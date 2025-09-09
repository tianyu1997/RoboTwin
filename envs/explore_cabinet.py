from ._base_task import Base_Task
from .utils import *
import sapien
import glob
import numpy as np

class explore_cabinet(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags, table_static=False)

    def rand_move(self, arm_tag, n=5, trans_limits=(0.03, 0.03, 0.02), rot_limit=0.05):
        """Perform n small random translations and small random quaternion rotations.

        trans_limits: tuple of (max_dx, max_dy, max_dz)
        rot_limit: maximum axis-angle rotation (radians) applied as +/- around a random axis
        """
        max_dx, max_dy, max_dz = trans_limits
        for _ in range(n):
            dx = np.random.uniform(-max_dx, max_dx)
            dy = np.random.uniform(-max_dy, max_dy)
            dz = np.random.uniform(-max_dz, max_dz)
            # small random axis-angle -> quaternion (x,y,z,w)
            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(-rot_limit, rot_limit)
            qw = float(np.cos(angle / 2.0))
            qxyz = (axis * np.sin(angle / 2.0)).astype(float)
            quat = [float(qxyz[0]), float(qxyz[1]), float(qxyz[2]), qw]
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy, z=dz, quat=quat))

    def load_actors(self):
        # create cabinet first
        self.model_name = "036_cabinet"
        self.model_id = 46653
        self.cabinet = rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-0.05, 0.05],
            ylim=[0.155, 0.155],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 1],
            fix_root_link=True,
        )

        # choose an object and place it inside the cabinet (use functional point 0 as base)
        object_list = [
            "047_mouse",
            "048_stapler",
            "057_toycar",
            "073_rubikscube",
            "075_bread",
            "077_phone",
            "081_playingcards",
            "112_tea-box",
            "113_coffee-box",
            "107_soap",
        ]
        self.selected_modelname = np.random.choice(object_list)

        # helper to find available model ids (reuse from above style)
        def get_available_model_ids(modelname):
            asset_path = os.path.join("assets/objects", modelname)
            json_files = glob.glob(os.path.join(asset_path, "model_data*.json"))
            available_ids = []
            for file in json_files:
                base = os.path.basename(file)
                try:
                    idx = int(base.replace("model_data", "").replace(".json", ""))
                    available_ids.append(idx)
                except ValueError:
                    continue
            return available_ids

        available_model_ids = get_available_model_ids(self.selected_modelname)
        if not available_model_ids:
            raise ValueError(f"No available model_data.json files found for {self.selected_modelname}")
        self.selected_model_id = np.random.choice(available_model_ids)

        # get a pose inside the cabinet's functional point and add a small random offset so object is inside
        func_pose = self.cabinet.get_functional_point(0, "pose")
        inside_p = np.array(func_pose.p)
        # small random offset within the cabinet footprint
        inside_p[0] += np.random.uniform(-0.03, 0.03)
        inside_p[1] += np.random.uniform(-0.03, 0.03)
        inside_p[2] += np.random.uniform(0.0, 0.03)
        pose = sapien.Pose(inside_p, func_pose.q)

        self.object = create_actor(
            scene=self,
            pose=pose,
            modelname=self.selected_modelname,
            convex=True,
            model_id=self.selected_model_id,
        )
        self.object.set_mass(0.01)

        # Record the cabinet functional point (drawer/door) initial world pose so we can
        # detect later whether it has been pulled open.
        try:
            fp = self.cabinet.get_functional_point(0, "pose")
            self.drawer_init_pose = fp
            self.drawer_init_p = np.array(fp.p)
        except Exception:
            # If functional point not available, mark as None and fallback in check
            self.drawer_init_pose = None
            self.drawer_init_p = None

        # prohibit collisions around cabinet and object
        self.add_prohibit_area(self.object, padding=0.01)
        self.add_prohibit_area(self.cabinet, padding=0.01)
        self.prohibited_area.append([-0.15, -0.3, 0.15, 0.3])

    def play_once(self):
        # Determine which arm to use based on object's x coordinate
        arm_tag = ArmTag('left')
        self.arm_tag = arm_tag
        self.origin_z = self.object.get_pose().p[2]

        # n次: choose a random count (kept similar to previous range)
        n = np.random.randint(5, 10)

        self.move(self.move_by_displacement(arm_tag=arm_tag , z=0.1))
        # First: perform n small random displacements with optional small random rotation
        self.rand_move(arm_tag=arm_tag, n=n)

        # Grasp and open the cabinet (use opposite arm for the cabinet handle)
        self.move(self.grasp_actor(self.cabinet, arm_tag=arm_tag , pre_grasp_dis=0.05))
        # pull/open the cabinet by displacing the opposite arm in -y several small steps
        for _ in range(4):
            self.move(self.move_by_displacement(arm_tag=arm_tag , y=-0.04))

        # release gripper using robot helper if available
        try:
            if str(arm_tag) == "left":
                self.robot.open_left_gripper()
            else:
                self.robot.open_right_gripper()
        except Exception:
            # fallback to Base_Task open_gripper if robot methods are not present
            self.open_gripper(arm_tag , pos=1.0)

        for _ in range(2):
            self.move(self.move_by_displacement(arm_tag=arm_tag , y=-0.04))

        # retreat to a safe / initial-like position: lift then move the arm away from the cabinet
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.15))
        back_x = -0.25 if str(arm_tag) == "right" else 0.25
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=back_x, y=0.2))

        # After opening, do another n small random displacements with small rotation
        self.rand_move(arm_tag=arm_tag, n=n)

        # populate info similarly to other tasks
        self.info["info"] = {
            "{A}": f"{self.selected_modelname}/base{self.selected_model_id}",
            "{B}": f"036_cabinet/base{0}",
            "{a}": str(arm_tag),
            "{b}": str(arm_tag ),
        }
        return self.info

    def check_success(self):
        # Determine success by whether the cabinet drawer/door functional point
        # has been pulled open relative to its initial recorded position.
        # if self.drawer_init_p is None:
        #     # fallback: if we didn't record the functional point, use original heuristic
        #     object_pose = self.object.get_pose().p
        #     target_pose = self.cabinet.get_functional_point(0)
        #     tag = np.all(abs(object_pose[:2] - target_pose[:2]) < np.array([0.07, 0.07]))
        #     height_ok = (object_pose[2] - self.origin_z) < 0.1
        #     gripper_open = (self.robot.is_left_gripper_open() if self.arm_tag == "left" else self.robot.is_right_gripper_open())
        #     return tag and height_ok and gripper_open

        try:
            cur_fp = self.cabinet.get_functional_point(0, "pose")
            cur_p = np.array(cur_fp.p)
        except Exception:
            return False

        # compute horizontal displacement in cabinet plane (XY)
        disp = np.linalg.norm(cur_p[:2] - self.drawer_init_p[:2])

        # threshold: consider opened if functional point moved more than 0.06 m
        opened = disp > 0.06

        return bool(opened)
