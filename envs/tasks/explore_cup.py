from .._base_task import Base_Task
from ..utils import *
import sapien
import glob
import numpy as np

class explore_cup(Base_Task):

    def setup_demo(self, **kwags):
        super()._init_task_env_(**kwags, table_static=False)


    def load_actors(self):
        # create cabinet first
        self.model_name = "036_cabinet"
        self.model_id = 46653
        self.cabinets = []
        x1 = 0.15
        x2 = 0.25
        y = 0
        self.cabinets.append(rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[x1, x2],
            ylim=[y, y],
            rotate_rand=False,
            rotate_lim=[0, 0, 0],
            qpos=[1, 0, 0, 1],
            fix_root_link=True,)
        )

        self.cabinets.append(rand_create_sapien_urdf_obj(
            scene=self,
            modelname=self.model_name,
            modelid=self.model_id,
            xlim=[-x2, -x1],
            ylim=[y, y],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 16],
            qpos=[1, 0, 0, 1],
            fix_root_link=True,)
        )

        # choose an object and place it inside the cabinet (use functional point 0 as base)
        object_list = [
            "071_can"
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
        n = np.random.randint(0,len(self.cabinets))
        self.x = 0.1 if n == 0 else -0.1
        self.cabinet = self.cabinets[n]
        func_pose = self.cabinet.get_functional_point(0, "pose")
        inside_p = np.array(func_pose.p)
        inside_p[0] += 2.2 * self.x  # move forward into the cabinet
        inside_p[1] += 0.08  # lift slightly above the functional point plane
        # small random offset within the cabinet footprint
        pose = sapien.Pose(inside_p, [0.5, 0.5, 0.5, 0.5])

        self.object = create_actor(
            scene=self,
            pose=pose,
            modelname=self.selected_modelname,
            convex=True,
            model_id=0,
        )
        self.object.set_mass(0.01)

        # Record the cabinet functional point (drawer/door) initial world pose so we can
        # detect later whether it has been pulled open.

        # prohibit collisions around cabinet and object
        self.add_prohibit_area(self.object, padding=0.01)
        for i in range(len(self.cabinets)):
            self.add_prohibit_area(self.cabinets[i], padding=0.01)
        self.prohibited_area.append([-0.15, -0.3, 0.15, 0.3])

    def play_once(self):
        # Determine which arm to use based on object's x coordinate
        flag = self.object.get_pose().p[0]
        arm_tag = ArmTag("right" if flag>0 else "left")
        self.arm_tag = arm_tag
        pose = self.get_arm_pose(arm_tag)
        self.origin_p = self.object.get_pose().p.copy()

        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=self.x))

        self.move(self.grasp_actor(self.object, arm_tag=arm_tag ,pre_grasp_dis=0.05))
        print("Grasped the object")

        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.075))

        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, y=-0.1))
    
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag , x=-self.x))

        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.075))

        self.open_gripper(arm_tag , pos=1.0)
        time.sleep(0.5)
        for _ in range(3):
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1))
        
        self.move(self.move_to_pose(arm_tag=arm_tag , target_pose=pose))

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

        

        # compute horizontal displacement in cabinet plane (XY)
        disp =np.linalg.norm(self.object.get_pose().p - self.origin_p)

        # threshold: consider opened if functional point moved more than 0.06 m
        lifted = disp > 0.1

        return lifted 
