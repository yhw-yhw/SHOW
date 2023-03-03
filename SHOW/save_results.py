from pathlib import Path
import trimesh
import numpy as np
import pyrender
import os
import PIL.Image as pil_img
import cv2
from loguru import logger

colors_dict = {
            'red': np.array([0.5, 0.2, 0.2]),
            'pink': np.array([0.7, 0.5, 0.5]),
            'neutral': np.array([0.7, 0.7, 0.6]),
            # 'purple': np.array([0.5, 0.5, 0.7]),
            'purple': np.array([0.55, 0.4, 0.9]),
            'green': np.array([0.5, 0.55, 0.3]),
            'sky': np.array([0.3, 0.5, 0.55]),
            'white': np.array([1.0, 0.98, 0.94]),
        }

def save_one_results(
    vertices,faces,
    
    img_size,#(height,width)
    center,#(cx,cy)
    focal_length,#(focalx,focaly)
    camera_pose,#K:(4,4)
    
    meta_data={},
    color_type='sky',
    input_renderer=None,
):

    save_smplpix_name=meta_data.get('save_smplpix_name',None)
    input_img=meta_data.get('input_img',None)
    output_name=meta_data.get('output_name',None)
    vertex_colors=meta_data.get('vertex_colors',None)
    obj_path=meta_data.get('obj_path',None)
    
    
    if save_smplpix_name is not None:
        out_mesh = trimesh.Trimesh(
            vertices, faces,
            vertex_colors=vertex_colors, 
            process=False)

        out_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0]))
        
        mesh = pyrender.Mesh.from_trimesh(out_mesh, smooth=False, wireframe=False)
        bg_color=[1.0, 1.0, 1.0, 0.0]
    else:
        color=colors_dict[color_type]
        
        out_mesh = trimesh.Trimesh(
            vertices, faces,
            process=False)

        out_mesh.apply_transform(trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0]))
        
        mesh = pyrender.Mesh.from_trimesh(
            out_mesh,

            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                roughnessFactor=0.6,
                alphaMode='OPAQUE',
                baseColorFactor=(color[0], color[1], color[2], 1.0)
            )
            )
        bg_color=[0.0, 0.0, 0.0, 0.0]
        
        if obj_path is not None:
            out_mesh.export(obj_path)
       
    if input_renderer is None:
        
        import platform
        if platform.system() == "Linux":
            os.environ['PYOPENGL_PLATFORM'] = 'egl'
        else:
            if 'PYOPENGL_PLATFORM' in os.environ:
                os.environ.__delitem__('PYOPENGL_PLATFORM')
            
        renderer = pyrender.OffscreenRenderer(
            viewport_width=img_size[1],
            viewport_height=img_size[0],
            point_size=1.0)
    else:
        renderer=input_renderer
        
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=(0.3, 0.3, 0.3))
    scene.add(mesh, 'mesh')


    camera = pyrender.camera.IntrinsicsCamera(
        fx=focal_length[0],fy=focal_length[1],
        cx=center[0],cy=center[1])
    scene.add(camera, pose=camera_pose)

    if 0:
        light_node = pyrender.DirectionalLight()
        scene.add(light_node, pose=camera_pose)
    else:
        light = pyrender.PointLight(color=np.array([1.0, 1.0, 1.0]) * 0.2, intensity=1)
        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose)

        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=15.0,
                        innerConeAngle=np.pi/3, outerConeAngle=np.pi/2)

        light_pose[:3, 3] = [1, 2, 2]
        scene.add(spot_l, pose=light_pose)

        light_pose[:3, 3] = [-1, 2, 2]
        scene.add(spot_l, pose=light_pose)

    color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    color = color.astype(np.float32) / 255.0

    if input_img!=None:
        if type(input_img)==str:
            # logger.info(f'load input_img:{input_img}')
            input_img=cv2.imread(input_img)
            input_img=cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
            
        assert(type(input_img)==np.ndarray)
        
        if input_img.max() > 1:
            input_img = input_img.astype(np.float32) / 255.0
            
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        output_img = (color[:, :, :-1] * valid_mask + (1 - valid_mask) * input_img)
        output_img = pil_img.fromarray((output_img * 255.).astype(np.uint8))
            
    if output_name!=None:
        Path(output_name).parent.mkdir(exist_ok=True,parents=True)
        output_img.save(output_name)
        # logger.info("save output_img:{}".format(output_name))

    if input_renderer is None:
        renderer.delete()
        
    return output_img
