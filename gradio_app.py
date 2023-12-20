import numpy as np
import gradio as gr
from FastSam_segmentation import segment_image
from llamaindex_bm25_baseline import ClassicRetrieverBM25
from upload_images_vector_store import image_similarity_activeloop

from utils import get_index_and_nodes_after_visual_similarity, load_vector_store
from global_variable import (
    VECTOR_STORE_PATH_IMAGES_MASKED,
    VECTOR_STORE_PATH_IMAGES_NORMAL,
    VECTOR_STORE_PATH_DESCRIPTION,
)
from PIL import Image
import cv2
from llama_index.retrievers import BM25Retriever
import urllib.parse

VESTOR_STORE_IMAGES_MASKED = None
VESTOR_STORE_IMAGES_NORMAL = None
VECTOR_STORE_DESCRIPTION = None


def search_from_image(input_image):
    print("----------- starting search from image -----------")
    iframe_html = '<iframe src={url} width="570px" height="400px"/iframe>'
    # iframe_html = """
    # <style>
    # .responsive-image {
    #     width: 100%; /* Imposta la larghezza dell'immagine al 100% del contenitore */
    #     max-width: 600px; /* Imposta una larghezza massima */
    #     min-width: 300px; /* Imposta una larghezza minima */
    #     height: auto; /* Mantiene il rapporto aspetto originale dell'immagine */
    # }
    # </style>

    # <iframe src={url} class="responsive-image" /iframe>"""

    iframe_url = f"https://app.activeloop.ai/visualizer/iframe?url={VECTOR_STORE_PATH_IMAGES_NORMAL}&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTY5ODY1NjM2NCwiZXhwIjoxNzYxODE0NzQxfQ.eyJpZCI6Im1hbnVmZSJ9.VkwsjjVroPcckI6CFl6XgFg88IaTnVqWdtXyA9AsmwFV6tgNSoKjIHM8Cl7oF2NwdNSsS30wKKT_71zLh9lWGg&query="

    global VESTOR_STORE_IMAGES_MASKED
    global VESTOR_STORE_IMAGES_NORMAL
    global VECTOR_STORE_DESCRIPTION
    desc = "Description pill "
    sd_eff = "Side-effects for the pill "
    if not VESTOR_STORE_IMAGES_MASKED:
        VESTOR_STORE_IMAGES_MASKED = load_vector_store(
            VECTOR_STORE_PATH_IMAGES_MASKED
        ).vectorstore
    if not VECTOR_STORE_DESCRIPTION:
        VECTOR_STORE_DESCRIPTION = load_vector_store(
            VECTOR_STORE_PATH_DESCRIPTION
        ).vectorstore
    # USED FOR NORMAL IMAGES RESEARCH
    # if not VESTOR_STORE_IMAGES_NORMAL:
    #     VESTOR_STORE_IMAGES_NORMAL = load_vector_store(
    #         VECTOR_STORE_PATH_IMAGES_NORMAL
    #     ).vectorstore

    image_path = "./test/input_image.png"
    image_path_masked = f"./test/{image_path.split('.')[0]}_masked.png"
    im = Image.fromarray(input_image)
    im.save(image_path)
    # cv2.imwrite(image_path, input_image)

    # MASK IMAGE
    image_masked = segment_image([image_path], test=True)
    image_masked = image_masked[0]
    image_masked_pil = Image.fromarray(image_masked)
    image_masked_pil.save(image_path_masked)
    cv2.imwrite(image_path_masked, image_masked)

    # VISUAL SIMILARITY
    similar_images = image_similarity_activeloop(VESTOR_STORE_IMAGES_MASKED, image_path)
    similar_images_after_segmentation = image_similarity_activeloop(
        VESTOR_STORE_IMAGES_MASKED, image_path_masked
    )
    filename_similar_images_to_retrieve_from_description_db = []
    for el in similar_images_after_segmentation["filename"]:
        filename_similar_images_to_retrieve_from_description_db.append(el)

    (
        _,
        nodes,
        _,
        filtered_elements,
    ) = get_index_and_nodes_after_visual_similarity(
        filename_similar_images_to_retrieve_from_description_db
    )  # node_0 is related to filtered_elements[0], ...

    # EXCLUDE THE 3 MOST SIMILAR IMAGES
    most_similar_3_images_filenames = [
        similar_images_after_segmentation["filename"][0],
        similar_images_after_segmentation["filename"][1],
        similar_images_after_segmentation["filename"][2],
    ]
    most_similar_3_node_ids = [
        filtered_elements["filename"].index(filename)
        for filename in most_similar_3_images_filenames
    ]
    nodes = [el for idx, el in enumerate(nodes) if idx not in most_similar_3_node_ids]
    # DESCRIPTION SIMILARITY
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    hybrid_only_bm25_retriever = ClassicRetrieverBM25(bm25_retriever)
    # most similar image (visually)
    id_most_similar = filtered_elements["filename"].index(
        similar_images_after_segmentation["filename"][0]
    )

    # description or the most similar image ==> use the id as key to retrieve the description
    description = filtered_elements["text"][id_most_similar]
    # ==> return sorted nodes based on description similarity
    nodes_bm25_response = hybrid_only_bm25_retriever.retrieve(description)

    # take the first 3 elements (most similar) given the description
    most_similar_id = most_similar_3_node_ids
    most_similar_images_filenames_metadata = [
        [
            filtered_elements["filename"][el],
            filtered_elements["metadata"][el],
            filtered_elements["text"][el],
        ]
        for el in most_similar_id
    ]
    output_name1 = most_similar_images_filenames_metadata[0][1]["name"]
    output_name2 = most_similar_images_filenames_metadata[1][1]["name"]
    output_name3 = most_similar_images_filenames_metadata[2][1]["name"]
    output_side_effects1 = most_similar_images_filenames_metadata[0][1]["side-effects"]
    output_side_effects2 = most_similar_images_filenames_metadata[1][1]["side-effects"]
    output_side_effects3 = most_similar_images_filenames_metadata[2][1]["side-effects"]
    output_description1 = most_similar_images_filenames_metadata[0][2]
    output_description2 = most_similar_images_filenames_metadata[1][2]
    output_description3 = most_similar_images_filenames_metadata[2][2]

    # most dissimilar element
    most_dissimilar_ids = nodes_bm25_response[-3:]
    most_dissimilar_id = [
        int(el.node_id.split("_")[1]) for el in most_dissimilar_ids
    ]  # i.e. from node_1, node_5, node_9 to 1, 5, 9
    most_dissimilar_images_filenames_metadata = [
        [
            filtered_elements["filename"][el],
            filtered_elements["metadata"][el],
            filtered_elements["text"][el],
        ]
        for el in most_dissimilar_id
    ]
    output_name4 = most_dissimilar_images_filenames_metadata[0][1]["name"]
    output_name5 = most_dissimilar_images_filenames_metadata[1][1]["name"]
    output_name6 = most_dissimilar_images_filenames_metadata[2][1]["name"]
    output_side_effects4 = most_dissimilar_images_filenames_metadata[0][1][
        "side-effects"
    ]
    output_side_effects5 = most_dissimilar_images_filenames_metadata[1][1][
        "side-effects"
    ]
    output_side_effects6 = most_dissimilar_images_filenames_metadata[2][1][
        "side-effects"
    ]
    output_description4 = most_dissimilar_images_filenames_metadata[0][2]
    output_description5 = most_dissimilar_images_filenames_metadata[1][2]
    output_description6 = most_dissimilar_images_filenames_metadata[2][2]

    # use the filename as key to retrieve the image and description given the most similar and dissimilar from the description
    filename_for_visualizer = [
        el[0]
        for el in most_similar_images_filenames_metadata
        + most_dissimilar_images_filenames_metadata
    ]
    id_most_similar_images = [
        similar_images_after_segmentation["filename"].index(el[0])
        for el in most_similar_images_filenames_metadata
    ]
    id_most_dissimilar_images = [
        similar_images_after_segmentation["filename"].index(el[0])
        for el in most_dissimilar_images_filenames_metadata
    ]

    most_similar_images = [
        similar_images_after_segmentation["image"][el] for el in id_most_similar_images
    ]
    most_dissimilar_images = [
        similar_images_after_segmentation["image"][el]
        for el in id_most_dissimilar_images
    ]
    images = [
        Image.fromarray(el) for el in most_similar_images + most_dissimilar_images
    ]
    query = "select image where filename == "
    # queries = [f"{query}'{el}'" for el in filename_for_visualizer]  # masked images
    queries = [
        f"""{query}'images/{el.split("/")[1].split("_masked")[0]}.jpg'"""
        for el in filename_for_visualizer
    ]  # normal images
    urls = [iframe_url + urllib.parse.quote(el) for el in queries]
    # url = iframe_url + urllib.parse.quote(query)
    # html = iframe_html.format(url=url)
    htmls = [iframe_html.format(url=url) for url in urls]

    return (
        htmls[0],
        htmls[1],
        htmls[2],
        htmls[3],
        htmls[4],
        htmls[5],
        gr.Textbox(label=f"{desc} {output_name1}", value=output_description1),
        gr.Textbox(label=f"{desc} {output_name2}", value=output_description2),
        gr.Textbox(label=f"{desc} {output_name3}", value=output_description3),
        gr.Textbox(label=f"{desc} {output_name4}", value=output_description4),
        gr.Textbox(label=f"{desc} {output_name5}", value=output_description5),
        gr.Textbox(label=f"{desc} {output_name6}", value=output_description6),
        gr.Textbox(label=f"{sd_eff} {output_name1}", value=output_side_effects1),
        gr.Textbox(label=f"{sd_eff} {output_name2}", value=output_side_effects2),
        gr.Textbox(label=f"{sd_eff} {output_name3}", value=output_side_effects3),
        gr.Textbox(label=f"{sd_eff} {output_name4}", value=output_side_effects4),
        gr.Textbox(label=f"{sd_eff} {output_name5}", value=output_side_effects5),
        gr.Textbox(label=f"{sd_eff} {output_name6}", value=output_side_effects6),
    )


with gr.Blocks(title="dwadqwdwq") as demo:
    gr.Markdown("# Compute the similarity between pills.")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("Upload a pill image.")
            image_input = gr.Image()
            image_button = gr.Button("Compute similarity")
        with gr.Column(scale=4):
            gr.Markdown("Most similar images:")
            with gr.Row():
                with gr.Column(scale=1):
                    # image_output1 = gr.Image()
                    image_output1 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc1 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects1 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output2 = gr.Image()
                    image_output2 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc2 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects2 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output3 = gr.Image()
                    image_output3 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc3 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects3 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
        with gr.Column(scale=4):
            gr.Markdown("Pay attention to these images:")
            with gr.Row():
                with gr.Column(scale=1):
                    # image_output4 = gr.Image()
                    image_output4 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc4 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects4 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output5 = gr.Image()
                    image_output5 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc5 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects5 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )
                with gr.Column(scale=1):
                    # image_output6 = gr.Image()
                    image_output6 = gr.HTML(
                        """
                       Loading Activeloop Visualizer ...
                        """
                    )
                    desc6 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Description pill",
                    )
                    side_effects6 = gr.Textbox(
                        lines=1,
                        scale=3,
                        label="Side-effects for the pill",
                    )

    image_button.click(
        search_from_image,
        inputs=image_input,
        outputs=[
            image_output1,
            image_output2,
            image_output3,
            image_output4,
            image_output5,
            image_output6,
            # NAME1,
            # NAME2,
            # NAME3,
            # NAME4,
            # NAME5,
            # NAME6,
            desc1,
            desc2,
            desc3,
            desc4,
            desc5,
            desc6,
            side_effects1,
            side_effects2,
            side_effects3,
            side_effects4,
            side_effects5,
            side_effects6,
        ],
    )

demo.launch()

# import gradio as gr


# # Define a function to process the input image and generate 6 different images
# def process_image(input_image):
#     # Your AI logic here to generate 6 different images from the input image
#     # You can replace the following line with your actual code
#     output_images = [input_image] * 6

#     return output_images


# # Create a Gradio interface
# iface = gr.Interface(
#     fn=process_image,
#     inputs="image",
#     outputs=gr.Tab("Flip Text"):
#         text_input = gr.Textbox()
#         text_output = gr.Textbox(),

#     live=True,
#     title="Image Processing",
#     description="Upload an image and get 6 different processed images as output.",
# )

# # Launch the Gradio interface
# iface.launch()


# import numpy as np
# import gradio as gr


# def flip_text(x):
#     return x[::-1]


# def flip_image(x):
#     return np.fliplr(x)


# with gr.Blocks() as demo:
#     gr.Markdown("Flip text or image files using this demo.")
#     with gr.Tab("Flip Text"):
#         text_input = gr.Textbox()
#         text_output = gr.Textbox()
#         text_button = gr.Button("Flip")
#     with gr.Tab("Flip Image"):
#         with gr.Row():
#             image_input = gr.Image()
#             image_output = gr.Image()
#         image_button = gr.Button("Flip")

#     with gr.Accordion("Open for More!"):
#         gr.Markdown("Look at me...")

#     text_button.click(flip_text, inputs=text_input, outputs=text_output)
#     image_button.click(flip_image, inputs=image_input, outputs=image_output)

# demo.launch()
