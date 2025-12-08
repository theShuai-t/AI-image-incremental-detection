from models.vit_xception_center import Vit_Xception_Center

def get_model(args):
    model_name = args["model_name"].lower()
    if model_name == 'vit_xception_center':
        return Vit_Xception_Center(args)
