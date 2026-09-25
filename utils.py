
from pathlib import Path

import plotly.express as px
from PIL import Image


def plot_frame(image_path: str=None):
    
    file_path = Path(image_path)
    if not file_path.is_file():
        raise ValueError("Image is not a file.")
    
    # Load frame
    frame = Image.open(image_path)

    fig = px.imshow(frame)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Hide axes
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()