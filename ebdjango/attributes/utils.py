import os
from tensorflow import keras
from random import shuffle

def determine_attributes(prediction):
    label_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                   'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                   'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                   'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    label_dict = {i:name for i,name in enumerate(label_names)}
    reverse_label_dict = {name:i for i, name in label_dict.items()}

    related = [
        ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'],
        ['Straight_Hair', 'Wavy_Hair']
    ]

    reinforcement = {
        '5_o_Clock_Shadow': .1,
        'Goatee': .1,
        'Mustache': .1,
        'Sideburns': .1,
        'Wearing_Necktie': .1,
        'Heavy_Makeup': -.1,
        'Wearing_Earrings': -.1,
        'Wearing_Lipstick': -.1,
        'Wearing_Necklace': -.1,
    }

    threshold = .3
    male = 20

    predition = prediction.tolist()
    print(prediction)
    intermediary = []

    # high threshold
    for value in prediction:
        if value < threshold:
            intermediary.append(0)
        elif value > 1 - threshold:
            intermediary.append(1)
        else:
            intermediary.append(value)

    # reinforce gender
    if intermediary[male] not in (1,0):
        for key in reinforcement:
            if intermediary[reverse_label_dict[key]] == 1:
                intermediary[male] += reinforcement[key]
    if intermediary[male] < threshold:
        intermediary[male] = 0
    elif intermediary[male] > 1 - threshold:
        intermediary[male] = 1

    # remove related if one is strong
    for d in related:
        print(d)
        if any([intermediary[reverse_label_dict[name]] == 1 for name in d]):
            for name in d:
                if name != 1:
                    del intermediary[reverse_label_dict[i]]

    results = {
        'sure': {'pos': [], 'neg': []},
        'unsure': {'pos': [], 'neg': []}
    }

    for i, value in enumerate(intermediary):
        name = label_dict[i]
        if value == 0:
            results['sure']['neg'].append(name)
        elif value == 1:
            results['sure']['pos'].append(name)
        elif value < .5:
            results['unsure']['neg'].append(name)
        else:
            results['unsure']['pos'].append(name)

    return results


def md(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)