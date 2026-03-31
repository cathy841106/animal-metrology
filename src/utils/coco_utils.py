
def get_category_ids_by_supercategory(coco, supercategories):
    """
    Get category IDs from COCO dataset based on given supercategories.
    
    Parameters:
        coco: COCO API object
        supercategories: list of supercategory names (e.g., ['animal'])
        
    Returns:
        List of category IDs that belong to the given supercategories
    """
    cat_ids = []
    for cat in coco.loadCats(coco.getCatIds()):
        if cat['supercategory'] in supercategories:
            cat_ids.append(cat['id'])
    return cat_ids

def get_image_ids_by_categories(coco, cat_ids):
    """
    Get image IDs that contain at least one object from given category IDs.
    
    Parameters:
        coco: COCO API object
        cat_ids: list of category IDs
        
    Returns:
        Set of image IDs containing at least one instance of the given categories
    """
    img_ids = set()
    for cat_id in cat_ids:
        img_ids.update(coco.getImgIds(catIds=[cat_id]))  # Get image ids for certain category id
    return img_ids

def filter_images_by_min_annotations(coco, img_ids, cat_ids, min_count=2):
    """
    Filter images that contain at least a minimum number of annotations for given categories.
    
    Parameters:
        coco: COCO API object
        img_ids: list of image IDs
        cat_ids: list of category IDs
        min_count: minimum number of annotations required (default=2)
        
    Returns:
        List of image IDs that satisfy the minimum annotation count
    """
    selected = []
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)  # Get annotation ids filtered by image ids and category ids
        if len(ann_ids) >= min_count:  # Just keep image that containing at least 2 animals
            selected.append(img_id)
    return selected

def build_file_to_imgid(coco):
    """
    Build a mapping from image file name to COCO image ID.
    
    Parameters:
        coco: COCO API object
        
    Returns:
        Dictionary mapping {file_name: image_id}
    """
    mapping = {}
    for img in coco.loadImgs(coco.getImgIds()):
        mapping[img["file_name"]] = img["id"]
    return mapping

def get_animal_cat_ids(coco):
    """
    Get category IDs for all categories whose supercategory is 'animal'.
    
    Parameters:
        coco: COCO API object
        
    Returns:
        List of category IDs belonging to 'animal' supercategory
    """
    return [
        cat['id']
        for cat in coco.loadCats(coco.getCatIds())
        if cat['supercategory'] == 'animal'
    ]

def get_gt_bboxes(coco, img_id, cat_ids):
    """
    Get ground truth bounding boxes for a specific image and categories.
    
    Parameters:
        coco: COCO API object
        img_id: image ID
        cat_ids: list of category IDs
        
    Returns:
        List of bounding boxes in COCO format [x, y, w, h]
    """
    ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(ann_ids)
    return [ann["bbox"] for ann in anns]  # xywh