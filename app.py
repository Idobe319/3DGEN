import os
import gradio as gr
from neoforge_core import preprocess_image, GeometryGenerator, process_retopology, generate_uvs, bake_texture

generator = GeometryGenerator()

def pipeline(image, poly_count, smooth_flow, im_iters=2, im_strength=0.5, do_bake=False, preview=False):
    working_dir = "workspace"
    os.makedirs(working_dir, exist_ok=True)
    input_path = os.path.join(working_dir, "input.png")
    image.save(input_path)
    clean_path = preprocess_image(input_path)
    raw_mesh = generator.generate(clean_path)
    clean_obj_name = os.path.join(working_dir, "clean_quads.obj")
    process_retopology(raw_mesh, clean_obj_name, vertex_count=int(poly_count), smooth_flow=bool(smooth_flow), im_iters=int(im_iters), im_strength=float(im_strength))
    final_obj_path, final_mesh = generate_uvs(clean_obj_name)
    if do_bake:
        baked = bake_texture(raw_mesh, final_obj_path)
        print('Baking result:', baked)
    # If preview requested, return Model3D and the file; otherwise hide preview
    if preview:
        return final_obj_path, final_obj_path
    else:
        # hide the Model3D component by returning None for it and the file path
        return None, final_obj_path

with gr.Blocks(title="NeoForge 3D Studio") as demo:
    gr.Markdown("# NeoForge 3D: AI to Professional Mesh")
    gr.Markdown("Create perfect quad-topology models from images locally.")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Upload Image")
            poly_slider = gr.Slider(minimum=500, maximum=10000, value=2500, step=100, label="Target Vertex Count")
            flow_check = gr.Checkbox(label="Enforce Edge Loops (Hard Surface mode)", value=True)
            im_iters = gr.Slider(minimum=0, maximum=5, step=1, value=2, label="Instant Meshes Extra Smooth Iterations")
            im_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Instant Meshes Smooth Strength")
            bake_check = gr.Checkbox(label="Bake Texture (Blender headless)", value=False)
            preview_check = gr.Checkbox(label="Enable interactive 3D preview (may cause WebGPU warnings)", value=False)
            run_btn = gr.Button("Generate Professional 3D Model", variant="primary")
        with gr.Column():
            # Model3D is optional because some browsers show WebGPU warnings; we only
            # create it when the preview checkbox is checked at runtime.
            output_model = gr.Model3D(label="Interactive Preview", visible=False)
            download_file = gr.File(label="Download OBJ")
    run_btn.click(
        fn=pipeline,
        inputs=[input_img, poly_slider, flow_check, im_iters, im_strength, bake_check, preview_check],
        outputs=[output_model, download_file]
    )

if __name__ == "__main__":
    demo.launch(share=True)
