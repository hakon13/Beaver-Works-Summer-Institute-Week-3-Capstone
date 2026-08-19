import gradio as gr
from typing import Callable, List, Tuple, Any

def create_frontend(search_function: Callable[[str, int], List[Tuple[Any, str, float]]]) -> gr.Blocks:
    dark_theme = gr.themes.Monochrome(
        primary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        body_text_color="*neutral_50",
        body_text_color_dark="*neutral_50",
        background_fill_secondary="*neutral_900",
        background_fill_secondary_dark="*neutral_900",
        border_color_primary="*neutral_800",
        border_color_primary_dark="*neutral_800",
    )

    with gr.Blocks(theme=dark_theme, title="CogWorks Image Search") as demo:
        gr.Markdown(
            """
            # Semantic Image Search
            Provide a caption to find an image from the COCO dataset 
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                query_input = gr.Textbox(
                    label="Search Query",
                    placeholder="horses on a beach...",
                    lines=1
                )
                
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Number of Results (k)"
                )
                
                search_btn = gr.Button("Search", variant="primary")

            with gr.Column(scale=2):
                results_gallery = gr.Gallery(
                    label="Search Results",
                    show_label=True,
                    elem_id="results-gallery",
                    columns=[2, 3], 
                    rows=[2],
                    object_fit="contain",
                    height="auto"
                )

        def process_search(query: str, k: int):
            if not query.strip():
                return []
            
            raw_results = search_function(query, k)
            
            formatted_gallery = []
            for item in raw_results:
                image_id, url, score = item
                label = f"ID: {image_id} | Score: {score:.3f}"
                formatted_gallery.append((url, label))
                
            return formatted_gallery

        search_btn.click(
            fn=process_search,
            inputs=[query_input, k_slider],
            outputs=[results_gallery]
        )
        
        query_input.submit(
            fn=process_search,
            inputs=[query_input, k_slider],
            outputs=[results_gallery]
        )

    return demo