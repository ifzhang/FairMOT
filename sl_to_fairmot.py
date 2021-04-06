import supervisely_lib as sly


workspace_id = 23821
project_id = 102425

convert_name = 'lemon'
gt_path = '/alex_work/gt_sl.txt'

api = sly.Api.from_env()
meta_json = api.project.get_meta(project_id)
meta = sly.ProjectMeta.from_json(meta_json)

datasets = api.dataset.get_list(project_id)
progress = sly.Progress("Start conversation", api.project.get_images_count(project_id))
for dataset in datasets:
    images = api.image.get_list(dataset.id)
    for batch in sly.batched(images):
        image_ids = [image_info.id for image_info in batch]
        ann_infos = api.annotation.download_batch(dataset.id, image_ids)
        frame_idx = 1
        for ann_info in ann_infos:
            ann = sly.Annotation.from_json(ann_info.annotation, meta)
            for idx, label in enumerate(ann.labels):
                if label.obj_class.name == convert_name:
                    left = label.geometry.left
                    top = label.geometry.top
                    width = label.geometry.right - label.geometry.left
                    height = label.geometry.bottom - label.geometry.top

                    gt = '{},{},{},{},{},{},{},{},{}\n'.format(frame_idx, (idx + 1), left, top, width, height, 1, -1, -1)
                    with open(gt_path, 'a') as f:
                        f.write(gt)

            frame_idx += 1

