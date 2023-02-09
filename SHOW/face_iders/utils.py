
from loguru import logger


def match_faces(img, face_ider, person_face_emb):
    # img: bgr,hw3,uint8
    faces = face_ider.get(img)
    if faces is None:
        return None, None
    # face_ider: 1.func:get(np_img) --> {2.normed_embedding,3.bbox}
    for face in faces:
        cur_emb = face.normed_embedding
        sim = face_ider.cal_emb_sim(cur_emb, person_face_emb)
        if sim >= face_ider.threshold:
            logger.info(f'found sim:{sim}')
            correspond_bbox = face.bbox
            xmin, ymin, xmax, ymax = correspond_bbox
            correspond_center = [
                int((xmin+xmax)/2),
                int((ymin+ymax)/2)]
            return correspond_center, correspond_bbox
        logger.info(f'not found: {sim}')
    return None, None