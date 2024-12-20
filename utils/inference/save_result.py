import json, os


def save_result(save_dir, data_id=0, garment_json=None, fig=None):
    os.makedirs(save_dir, exist_ok=True)
    garment_dir = os.path.join(save_dir, f"garment_" + f"{data_id}".zfill(5) )
    os.makedirs(garment_dir, exist_ok=True)
    save_path = os.path.join(garment_dir, f"garment"+ ".json")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(garment_json, f, indent=4)

    if fig is not None:
        fig.write_html(os.path.join(garment_dir,"vis_comp.html"))
