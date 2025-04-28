import gradio as gr
from datetime import datetime
from hybrid_processor import HybridProcessor
import logging
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    processor = HybridProcessor()
except Exception as e:
    logger.error(f"Failed to initialize HybridProcessor: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def compare_approaches(question, processor):
    try:
        vector_result = processor.vector_processor.query(question)
        graph_result = processor.graph_processor.query(question)
        hybrid_result = processor.query(question)

        vector_chunks_str = '\n\n'.join(
            [f'Chunk {i + 1}: {chunk}' for i, chunk in enumerate(hybrid_result['chunks']['vector'])])
        graph_chunks_str = '\n\n'.join(
            [f'Record {i + 1}: {chunk}' for i, chunk in enumerate(hybrid_result['chunks']['graph'])])
        hybrid_chunks_formatted = f"Vector Chunks:\n{vector_chunks_str}\n\nGraph Records:\n{graph_chunks_str}"

        return {
            "vector_answer": vector_result["answer"],
            "vector_chunks": "\n\n".join(
                [f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(vector_result["chunks"])]),
            "graph_answer": graph_result["answer"],
            "graph_chunks": "\n\n".join([f"Record {i + 1}: {chunk}" for i, chunk in enumerate(graph_result["chunks"])]),
            "hybrid_answer": hybrid_result["answer"],
            "hybrid_chunks": hybrid_chunks_formatted
        }
    except Exception as e:
        logger.error(f"Error in compare_approaches: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "vector_answer": "Error processing query.",
            "vector_chunks": "",
            "graph_answer": "Error processing query.",
            "graph_chunks": "",
            "hybrid_answer": "Error processing query.",
            "hybrid_chunks": ""
        }

def chat_function(message, history):
    try:
        response = compare_approaches(message, processor)
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant",
             "content": f"**Vector DB Answer**: {response['vector_answer']}\n\n**Graph DB Answer**: {response['graph_answer']}\n\n**Hybrid Answer**: {response['hybrid_answer']}"}
        ]
        return (
            new_history,
            "",
            response["vector_chunks"],
            response["graph_chunks"],
            response["hybrid_chunks"]
        )
    except Exception as e:
        logger.error(f"Error in chat_function: {str(e)}")
        logger.error(traceback.format_exc())
        return history, "", f"Error: {str(e)}", "", ""

def upload_file(file):
    try:
        if file is None:
            return "Please upload a PDF file"
        filename = processor.process_and_store_pdf(file)
        return f"Processed: {filename}"
    except Exception as e:
        logger.error(f"Error in upload_file: {str(e)}")
        logger.error(traceback.format_exc())
        return f"Error processing file: {str(e)}"

modern_theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
    text_size="md",
)

custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .container { max-width: 1200px; margin: 0 auto; }
    .header { text-align: center; padding: 1rem 0; }
    .chatbot .message { border-radius: 10px; padding: 10px; margin: 5px 0; }
    .user-message { background-color: #E94560 !important; color: white; }
    .assistant-message { background-color: #0F3460 !important; color: white; }
    .message-input { background-color: #0F3460; color: white; border: 1px solid #E94560; border-radius: 8px; }
    .message-input textarea { height: 60px !important; }
    .send-btn { background-color: #E94560; color: white; border: none; height: 60px !important; display: flex; align-items: center; justify-content: center; font-size: 18px !important; font-weight: 500; }
    .send-btn:hover { background-color: #FF6B6B; }
    .file-upload { background-color: #16213E; border: 1px dashed #E94560; border-radius: 8px; height: 200px; }
    .status-text { color: #E94560; font-size: 14px; text-align: center; }
    .info-box { background-color: #16213E; color: white; border: 1px solid #0F3460; border-radius: 8px; }
    .gr-button { transition: all 0.3s ease; }
    .gr-accordion { background-color: #16213E; border: 1px solid #0F3460; }
    .square-column { min-width: 300px; max-width: 400px; }
    .checkbox-group { max-height: 150px; overflow-y: auto; }
    .gr-accordion > div > button { font-size: 18px !important; font-weight: 500; }
    .header-text { color: white; font-size: 16px; margin-top: -10px; }
    .warning-text { color: #E94560; font-size: 16px; margin-top: -10px; }
"""

try:
    with gr.Blocks(theme=modern_theme, css=custom_css) as demo:
        gr.Markdown(
            """
            # Document Query Comparison
            <p class='header-text'>Compare Vector DB, Graph DB, and Hybrid approaches for document querying.</p>
            <p class='warning-text'><b>Do not upload personal or private data</b></p>
            """,
            elem_classes="header"
        )

        with gr.Row(elem_classes="container"):
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label=None,
                    type="messages",
                    height=500,
                    elem_classes="info-box"
                )
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Ask in the format: 'What is the relationship between [Entity1] and [Entity2]?' (e.g., 'What is the relationship between Alice and TechCorp?')",
                        show_label=False,
                        container=False,
                        elem_classes="message-input",
                        lines=3,
                        scale=8
                    )
                    send_btn = gr.Button(
                        "Send",
                        size="sm",
                        elem_classes="send-btn",
                        scale=2
                    )

            with gr.Column(scale=3):
                with gr.Accordion("File Upload", open=False):
                    file_upload = gr.File(
                        label="Drop PDF here",
                        file_types=[".pdf"],
                        elem_classes="file-upload",
                        height=200
                    )
                    upload_btn = gr.Button(
                        "Upload PDF",
                        size="sm"
                    )
                    upload_status = gr.Textbox(
                        value="Upload a PDF to begin",
                        show_label=False,
                        interactive=False,
                        elem_classes="status-text"
                    )

                with gr.Accordion("Manage Documents", open=False):
                    try:
                        initial_files = sorted(processor.get_processed_files())
                        logger.info(f"Initialized files_selection with {len(initial_files)} files: {initial_files}")
                    except Exception as e:
                        logger.error(f"Error fetching processed files: {str(e)}")
                        logger.error(traceback.format_exc())
                        initial_files = []
                    files_selection = gr.CheckboxGroup(
                        choices=initial_files,
                        label="Document Selection (Select one)",
                        interactive=True,
                        elem_classes="info-box checkbox-group"
                    )
                    delete_btn = gr.Button(
                        "Delete Document",
                        size="sm"
                    )
                    vector_chunks_display = gr.Textbox(
                        value="Vector DB results will appear here",
                        label="Vector DB Chunks",
                        interactive=False,
                        lines=5,
                        max_lines=5,
                        elem_classes="info-box"
                    )
                    graph_chunks_display = gr.Textbox(
                        value="Graph DB results will appear here",
                        label="Graph DB Records",
                        interactive=False,
                        lines=5,
                        max_lines=5,
                        elem_classes="info-box"
                    )
                    hybrid_chunks_display = gr.Textbox(
                        value="Hybrid results will appear here",
                        label="Hybrid Chunks/Records",
                        interactive=False,
                        lines=5,
                        max_lines=5,
                        elem_classes="info-box"
                    )

        gr.Markdown(
            """
            ---
            Powered by xAI
            """,
            elem_classes="footer"
        )

        def update_selection(selected_files):
            try:
                if len(selected_files) > 1:
                    return [selected_files[-1]]
                return selected_files
            except Exception as e:
                logger.error(f"Error in update_selection: {str(e)}")
                logger.error(traceback.format_exc())
                return []

        send_btn.click(
            chat_function,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, vector_chunks_display, graph_chunks_display, hybrid_chunks_display]
        )
        msg.submit(
            chat_function,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg, vector_chunks_display, graph_chunks_display, hybrid_chunks_display]
        )
        upload_btn.click(
            upload_file,
            inputs=[file_upload],
            outputs=[upload_status]
        ).then(
            lambda: gr.update(choices=sorted(processor.get_processed_files())),
            outputs=[files_selection]
        ).then(
            lambda: [],
            outputs=[files_selection]
        )

        files_selection.change(
            update_selection,
            inputs=[files_selection],
            outputs=[files_selection]
        )

        delete_btn.click(
            processor.delete_file_points,
            inputs=[files_selection],
            outputs=[upload_status, files_selection]
        ).then(
            lambda: [],
            outputs=[files_selection]
        )

except Exception as e:
    logger.error(f"Error initializing Gradio interface: {str(e)}")
    logger.error(traceback.format_exc())
    raise

if __name__ == "__main__":
    try:
        logger.info(f"===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
        demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, debug=True)
    except Exception as e:
        logger.error(f"Error launching Gradio app: {str(e)}")
        logger.error(traceback.format_exc())
        raise