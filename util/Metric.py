import util.Box as Box


class Metric:

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def eval(self, gtbox_list, predbox_list):

        if gtbox_list is None:
            raise Exception("empty gtbox list")

        # its okay for predbox list to be empty because we can still calculate
        # precision and recall.

        # assign closest gt for each predbox if the iou exceeds threshold
        predbox_map = {}
        nogt_predbox_list = []

        for box in predbox_list:

            maxiou = 0.0
            maxbox = None
            for gbox in gtbox_list:
                iou = Box.calculate_iou(box, gbox)
                if iou > self.threshold and iou > maxiou:
                    maxiou = iou
                    maxbox = gbox

                if maxbox is not None:
                    predbox_map[box] = maxbox
                else:
                    nogt_predbox_list.append(box)

        # count pred boxes with gtbox assigned (counting TP)
        TP = len(predbox_map)

        # count pred boxes with no gtbox assigned
        FP = len(nogt_predbox_list)

        # count gts that we missed
        detected_gts = list(predbox_map.values())

        no_detected_gts = [
            item for item in gtbox_list if item not in detected_gts]

        FN = len(no_detected_gts)

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        return precision, recall
