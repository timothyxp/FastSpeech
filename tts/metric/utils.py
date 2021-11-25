import editdistance


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text) == 0:
        return 0

    return editdistance.distance(predicted_text, target_text) / len(target_text)


def calc_wer(target_text, predicted_text) -> float:
    target_text, predicted_text = target_text.split(' '), predicted_text.split(' ')
    if len(target_text) == 0:
        return 0

    return editdistance.distance(predicted_text, target_text) / len(target_text)
