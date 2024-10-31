def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    of_x = max(bbox1[0], bbox2[0])
    of_y = max(bbox1[1], bbox2[1])
    os_x = min(bbox1[2], bbox2[2])
    os_y = min(bbox1[3], bbox2[3])
    
    inter = 0
    if of_x < os_x and of_y < os_y:
        inter += (os_x-of_x ) * (os_y-of_y)
    overlap = (bbox1[2]-bbox1[0]) * (bbox1[3] -bbox1[1])+ (bbox2[2]-bbox2[0]) * (bbox2[3] -bbox2[1]) - inter
    return inter / overlap



def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj = {obj[0]: obj[1:] for obj in frame_obj}
        frame_hyp = {hyp[0]: hyp[1:] for hyp in frame_hyp}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        delete = {}
        for obj_id, hyp_id in matches.items():
            if obj_id in frame_obj.keys() and hyp_id in frame_hyp.keys() \
            and iou_score(frame_obj[obj_id],frame_hyp[hyp_id]) > threshold:
                dist_sum += iou_score(frame_obj[obj_id],frame_hyp[hyp_id])
                match_count+=1
                delete[obj_id] = hyp_id
        for o, h in delete.items():
            del frame_obj[o]
            del frame_hyp[h]
            

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        new_matches = []
        for obj_id in frame_obj.keys():
            for hyp_id in frame_hyp.keys():
                iou = iou_score(frame_obj[obj_id],frame_hyp[hyp_id])
                if iou> threshold:
                    new_matches.append([iou, obj_id, hyp_id])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_matches = sorted(new_matches, key=lambda x:x[0], reverse=True)
        for pair in new_matches:
            if pair[1] in frame_obj.keys() and pair[2] in frame_hyp.keys() and pair[0]>threshold:
                dist_sum += pair[0]
                match_count+=1
                del frame_obj[pair[1]]
                del frame_hyp[pair[2]]
                matches[pair[1]] = pair[2]
                

        # Step 5: Update matches with current matched IDs
        pass

    # Step 6: Calculate MOTP
    MOTP = dist_sum/match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    gt_num = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        frame_obj = {obj[0]: obj[1:] for obj in frame_obj}
        frame_hyp = {hyp[0]: hyp[1:] for hyp in frame_hyp}
        gt_num+=len(frame_obj)
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        delete = {}
        for obj_id, hyp_id in matches.items():
            if obj_id in frame_obj.keys() and hyp_id in frame_hyp.keys() \
            and iou_score(frame_obj[obj_id],frame_hyp[hyp_id]) > threshold:
                dist_sum += iou_score(frame_obj[obj_id],frame_hyp[hyp_id])
                match_count+=1
                delete[obj_id] = hyp_id
        for o, h in delete.items():
            del frame_obj[o]
            del frame_hyp[h]
            
        
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        new_matches = []
        for obj_id in frame_obj.keys():
            for hyp_id in frame_hyp.keys():
                iou = iou_score(frame_obj[obj_id],frame_hyp[hyp_id])
                if iou> threshold:
                    new_matches.append([iou, obj_id, hyp_id])

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        new_matches = sorted(new_matches, key=lambda x:x[0], reverse=True)
        for pair in new_matches:
            if pair[1] in frame_obj.keys() and pair[2] in frame_hyp.keys() and pair[0]>threshold:
                dist_sum += pair[0]
                match_count+=1
                del frame_obj[pair[1]]
                del frame_hyp[pair[2]]
                if pair[1] in matches.keys() and matches[pair[1]] != pair[2]:
                    mismatch_error +=1
                matches[pair[1]] = pair[2]
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error

        # Step 6: Update matches with current matched IDs

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        false_positive += len(frame_hyp)
        missed_count += len(frame_obj)
        pass

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum/match_count
    MOTA = 1 - (missed_count+false_positive+mismatch_error)/gt_num

    return MOTP, MOTA
