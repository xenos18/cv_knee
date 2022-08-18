import dash
import dash_bootstrap_components as dbc
import base64
import numpy as np
import io
import cv2
import torch

from model import MRINet
from copy import deepcopy
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from dash import dcc
from dash import html
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import CAM, ScoreCAM, SSCAM, ISCAM, GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM
from torchcam.utils import overlay_mask

device = "cuda" if torch.cuda.is_available() else "cpu"

model_planes = {
    "axial": torch.load("./models/axial.pth"),
    "coronal": torch.load("./models/coronal.pth"),
    "sagittal": torch.load("./models/sagittal.pth")
}

image_planes = {"axial": None, "coronal": None, "sagittal": None}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MORPH])
server = app.server

app.layout = html.Div(
    [
        html.H1("RESULTS OF INTERPRETATION OF MODELS OF CLASSIFICATION OF MRI IMAGES", style={'text-align': 'center'}),
        html.Br(),
        html.H3("Load MRI images according to the specified planes:", style={'text-align': 'center'}),
        html.Br(),
        dbc.Row([
            dbc.Col(
                dcc.Upload(
                    id="load_axial_data",
                    children=[
                        dbc.Card(
                            [
                                dbc.CardImg(id="axial_data", src=None, top=True),
                                dbc.CardBody(
                                    html.P("Axial plane", style={'text-align': 'center'})
                                ),
                            ]
                        )
                    ],
                    multiple=False,
                ),
            ),
            dbc.Col([
                dcc.Upload(
                    id="load_coronal_data",
                    children=[
                        dbc.Card(
                            [
                                dbc.CardImg(id="coronal_data", src=None, top=True),
                                dbc.CardBody(
                                    html.P("Сoronal plane", style={'text-align': 'center'})
                                ),
                            ]
                        )
                    ],
                    multiple=False,
                ),
                html.Br(),
                dbc.Button(
                    "Diagnose",
                    id='predict_button',
                    color="primary",
                    n_clicks=0,
                    style={
                        "width": "100%"
                    }
                )],
            ),
            dbc.Col(
                dcc.Upload(
                    id="load_sagittal_data",
                    children=[
                        dbc.Card(
                            [
                                dbc.CardImg(id="sagittal_data", src=None, top=True),
                                dbc.CardBody(
                                    html.P("Sagittal plane", style={'text-align': 'center'})
                                ),
                            ]
                        )
                    ],
                    multiple=False,
                ),
            ),
        ]),
        html.Br(),
        dbc.Row(id="diagnose", children=[], justify="center"),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Label('Select CAM algorithm for interpretation:'),
                dcc.Dropdown(
                    id='choose_cams',
                    options=[
                        {"label": "CAM", "value": "CAM"},
                        {"label": "ScoreCAM", "value": "ScoreCAM"},
                        {"label": "SSCAM", "value": "SSCAM"},
                        {"label": "ISCAM", "value": "ISCAM"},
                        {"label": "GradCAM", "value": "GradCAM"},
                        {"label": "GradCAMpp", "value": "GradCAMpp"},
                        {"label": "SmoothGradCAMpp", "value": "SmoothGradCAMpp"},
                        {"label": "XGradCAM", "value": "XGradCAM"},
                        {"label": "LayerCAM", "value": "LayerCAM"},
                    ],
                    value="CAM",
                    clearable=False,
                    searchable=True,
                    multi=False,
                    style={"textAlign": "center"}
                ),
            ])
        ]),
        html.Br(),
        dbc.Row(id="cams", children=[
            dbc.Col([
                dbc.Row([
                    dcc.Slider(min=1, max=10, value=None, id='axial_slider',
                               tooltip={"placement": "bottom", "always_visible": True})
                ]),
                dbc.Row([
                    dbc.Col(html.Img(id="axial_img")),
                    dbc.Col(html.Img(id="axial_cam_img"))
                ], justify="center"),
            ]),
            dbc.Col([
                dbc.Row([
                    dcc.Slider(min=1, max=10, value=None, id='coronal_slider',
                               tooltip={"placement": "bottom", "always_visible": True})
                ]),
                dbc.Row([
                    dbc.Col(html.Img(id="coronal_img")),
                    dbc.Col(html.Img(id="coronal_cam_img"))
                ], justify="center"),
            ]),
            dbc.Col([
                dbc.Row([
                    dcc.Slider(min=1, max=10, value=None, id='sagittal_slider',
                               tooltip={"placement": "bottom", "always_visible": True})
                ]),
                dbc.Row([
                    dbc.Col(html.Img(id="sagittal_img"), ),
                    dbc.Col(html.Img(id="sagittal_cam_img"))
                ], justify="center"),
            ]),
        ], justify="center"),
    ]
)


def to_png(npy_img: np.array, number_slices: int = None) -> str:
    if number_slices is not None and npy_img.shape[0] > number_slices > 0:
        npy_img = npy_img[number_slices]
    is_success, buffer = cv2.imencode(".png", npy_img)
    io_buf = io.BytesIO(buffer)
    encoded_image = base64.b64encode(io_buf.getvalue())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


def parse_images(content: str, plane: str) -> str:
    global image_planes
    content_type, content_string = content.split(",")
    decoded = base64.b64decode(content_string)
    npy_img = np.load(io.BytesIO(decoded))
    image_planes[plane] = npy_img
    img = image_planes[plane]
    return to_png(img, img.shape[0] // 2)


def predict_diagnosis() -> object:
    images = {
        "axial": to_tensor(image_planes["axial"]),
        "coronal": to_tensor(image_planes["coronal"]),
        "sagittal": to_tensor(image_planes["sagittal"])
    }

    probas = np.array([0.0, 0.0, 0.0])
    for plane in ["axial", "coronal", "sagittal"]:
        input_tensor = normalize(resize(images[plane], (224, 224)) / 255.,
                                 [0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        with torch.no_grad():
            model = deepcopy(model_planes[plane]).to(device)
            model.eval()
            prediction = model(input_tensor.float())
        probas += torch.squeeze(torch.sigmoid(prediction).cpu().detach(), 0).numpy()

    probas = probas / 3
    return html.H4(f"С вероятностью {probas[0]:0.2f} данный пациент имеет разрыв передней крестообразной связки."
                   f" Вероятность травмы мениска равна {probas[1]:0.2f}."
                   f" Вероятность наличия аномалии коленного сустава равна {probas[2]:0.2f}.",
                   style={'text-align': 'center'})


def to_tensor(img_npy: np.array) -> torch.tensor:
    image = torch.FloatTensor(img_npy)
    image = image.repeat(3, 1, 1, 1).permute(1, 0, 2, 3)
    image = torch.squeeze(image, dim=0).to(device)
    return image


def get_cam_extractor(model, cam_mode: str):
    cam_extractors = {
        "CAM": CAM,
        "ScoreCAM": ScoreCAM,
        "SSCAM": SSCAM,
        "ISCAM": ISCAM,
        "GradCAM": GradCAM,
        "GradCAMpp": GradCAMpp,
        "SmoothGradCAMpp": SmoothGradCAMpp,
        "XGradCAM": XGradCAM,
        "LayerCAM": LayerCAM
    }

    return cam_extractors[cam_mode](model, model.layer4)


def to_cam(img_npy: np.array, slices: int, model, cam_mode) -> str:
    model = deepcopy(model).to(device)
    model.eval()
    cam_extractor = get_cam_extractor(model.base_model, cam_mode)

    img = to_tensor(img_npy)
    img = img[slices]
    img = img.reshape(1, *img.shape)

    input_tensor = normalize(resize(img, (224, 224)) / 255.,
                             [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

    out = model(input_tensor.float())

    try:
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)
    except Exception:
        return to_png(img_npy, slices)

    fused_cam = cam_extractor.fuse_cams(cams)

    cam_img = overlay_mask(to_pil_image(img[0]), to_pil_image(fused_cam, mode='F'), alpha=0.3)

    cam_extractor.clear_hooks()
    model.zero_grad()

    io_buf = io.BytesIO()
    cam_img.save(io_buf, format="PNG")
    encoded_image = base64.b64encode(io_buf.getvalue())
    return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(
    Output("axial_data", "src"),
    [
        Input("load_axial_data", "contents"),
        Input("load_axial_data", "filename")
    ],
)
def upload_axial_plane(contents, filename):
    if contents:
        return parse_images(contents, "axial")
    raise PreventUpdate


@app.callback(
    Output("coronal_data", "src"),
    [
        Input("load_coronal_data", "contents"),
        Input("load_coronal_data", "filename")
    ],
)
def upload_coronal_plane(contents, filename):
    if contents:
        return parse_images(contents, "coronal")
    raise PreventUpdate


@app.callback(
    Output("sagittal_data", "src"),
    [
        Input("load_sagittal_data", "contents"),
        Input("load_sagittal_data", "filename")
    ],
)
def upload_sagittal_plane(contents, filename):
    if contents:
        return parse_images(contents, "sagittal")
    raise PreventUpdate


@app.callback(
    Output("diagnose", "children"),
    Input("predict_button", "n_clicks")
)
def make_diagnose(n_clicks):
    if n_clicks > 0:
        return predict_diagnosis()
    raise PreventUpdate


@app.callback(
    [
        Output("axial_slider", "max"),
        Output("axial_slider", "value"),
        Output("coronal_slider", "max"),
        Output("coronal_slider", "value"),
        Output("sagittal_slider", "max"),
        Output("sagittal_slider", "value"),
    ],
    Input('predict_button', 'n_clicks'),
)
def update_sliders(n_clicks):
    if n_clicks > 0:
        max_axial = image_planes["axial"].shape[0] - 1
        max_coronal = image_planes["coronal"].shape[0] - 1
        max_sagittal = image_planes["sagittal"].shape[0] - 1
        return [
            max_axial,
            max_axial // 2,
            max_coronal,
            max_coronal // 2,
            max_sagittal,
            max_sagittal // 2,
        ]
    raise PreventUpdate


@app.callback(
    [
        Output("axial_img", "src"),
        Output("axial_cam_img", "src")
    ],
    [
        Input("axial_slider", "value"),
        Input("choose_cams", "value"),
    ],
)
def update_axial_cams(slices, cam_mode):
    if slices is None:
        raise PreventUpdate

    return [
        to_png(image_planes["axial"], slices),
        to_cam(image_planes["axial"], slices, model_planes["axial"], cam_mode)
    ]


@app.callback(
    [
        Output("coronal_img", "src"),
        Output("coronal_cam_img", "src")
    ],
    [
        Input("coronal_slider", "value"),
        Input("choose_cams", "value")
    ],
)
def update_coronal_cams(slices, cam_mode):
    if slices is None:
        raise PreventUpdate

    return [
        to_png(image_planes["coronal"], slices),
        to_cam(image_planes["coronal"], slices, model_planes["coronal"], cam_mode)
    ]


@app.callback(
    [
        Output("sagittal_img", "src"),
        Output("sagittal_cam_img", "src")
    ],
    [
        Input("sagittal_slider", "value"),
        Input("choose_cams", "value")
    ],
)
def update_sagittal_cams(slices, cam_mode):
    if slices is None:
        raise PreventUpdate

    return [
        to_png(image_planes["sagittal"], slices),
        to_cam(image_planes["sagittal"], slices, model_planes["sagittal"], cam_mode)
    ]


if __name__ == "__main__":
    app.run_server(debug=True)
