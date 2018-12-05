import os
import numpy as np
from tensorflow import keras
from random import shuffle

def determine_attributes(prediction):
    label_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
                   'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                   'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                   'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                   'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
                   'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    has = [' has ', ' does not have ']
    hasa = [' has a ', 'does not have a']
    _is = [' is ', ' is not ']
    pretty_print = {
        '5_o_Clock_Shadow': has,
        'Arched_Eyebrows': has,
        'Attractive': _is,
        'Bags_Under_Eyes': has,
        'Bald': _is,
        'Bangs': has,
        'Big_Lips': has,
        'Big_Nose': hasa,
        'Black_Hair': has,
        'Blond_Hair': has,
        'Blurry': _is,
        'Brown_Hair': has,
        'Bushy_Eyebrows': has,
        'Chubby': _is,
        'Double_Chin': hasa,
        'Eyeglasses': has,
        'Goatee': hasa,
        'Gray_Hair': has,
        'Heavy_Makeup': has,
        'High_Cheekbones': has,
        'Male': _is,
        'Mouth_Slightly_Open': hasa,
        'Mustache': hasa,
        'Narrow_Eyes': has,
        'Oval_Face': hasa,
        'Pale_Skin': has,
        'Pointy_Nose': hasa,
        'Receding_Hairline': hasa,
        'Rosy_Cheeks': has,
        'Sideburns': has,
        'Smiling': _is,
        'Straight_Hair': has,
        'Wavy_Hair': has,
        'Wearing_Earrings': _is,
        'Wearing_Hat': _is,
        'Wearing_Lipstick': _is,
        'Wearing_Necklace': _is,
        'Wearing_Necktie': _is,
        'Young': _is
    }

    No_Beard = ['does not have a beard', 'has a bearc']

    accuracy_under_headers = { # one standard deviation on ones scores
        'Arched_Eyebrows': 0.18,
        'Bags_Under_Eyes': 0.15,
        'Big_Lips': 0.09,
        'Big_Nose': 0.20,
        'Black_Hair': 0.19,
        'Brown_Hair': 0.13,
        'Oval_Face': 0.11,
        'Pointy_Nose': 0.15,
        'Straight_Hair': 0.06,
        'Wavy_Hair': 0.15,
        'Wearing_Earrings': 0.14,
        'Wearing_Necklace': 0.07
    }

    non_adjustable_under_headers = [] #todo

    predition_threshold = .5
    uncertainty_threshold = .4
    label_dict = {i:name for i,name in enumerate(label_names)}
    reverse_label_dict = {name:i for i, name in label_dict.items()}
    predition = prediction.tolist()

    # initial adjustment (st dev on under accuracy headers)
    adjusted_prediction = []
    accuracy_adjusted = [False]*len(prediction)

    for index, value in enumerate(prediction):
        if label_dict[index] in accuracy_under_headers:
            if value < predition_threshold:
                adjusted_prediction.append(value + accuracy_under_headers[label_dict[index]])
                accuracy_adjusted[index] = True
            else:
                adjusted_prediction.append(value)
        else:
            adjusted_prediction.append(value)

    # related adjustment
    related = [
        {'Black_Hair': 0.19, 'Blond_Hair': 0.25, 'Brown_Hair': 0.13, 'Gray_Hair': 0.18},
        {'Straight_Hair': 0.06, 'Wavy_Hair': 0.15}
    ]

    for d in related:
        keys = list(d.keys())
        print(keys)
        # values = [d[k] for k in keys]
        l = [adjusted_prediction[reverse_label_dict[name]] for name in keys]
        i = np.argmax(np.array(l))
        rel_key = keys[i]
        if adjusted_prediction[reverse_label_dict[rel_key]] < predition_threshold:
            adjusted_prediction[reverse_label_dict[rel_key]] += d[rel_key]

        print("STRONGEST")
        print(rel_key)

    # gender adjustment
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
    male = reverse_label_dict['Male']
    print("MALE")
    print(male)
    # if adjusted_prediction[male] < predition_threshold:
    for key in reinforcement:
        if adjusted_prediction[reverse_label_dict[key]] >= predition_threshold:
            print("ADJUSTING MALE")
            adjusted_prediction[male] += reinforcement[key]
            accuracy_adjusted[male] = True

    results_as_strings = []

    for index, value in enumerate(adjusted_prediction):
        # if value < .2:
        #     attribute = label_dict[index]

        #     if attribute == 'No_Beard':
        #         desc = No_Beard[1]
        #         res = 'The person probably {}'.format(desc)
        #     else:
        #         desc = pretty_print[attribute][1]
        #         res = 'The person probably {} {}'.format(desc, attribute.replace('_', ' ').lower())
        #     results_as_strings.append(res)
        if value > .5:
            attribute = label_dict[index]

            if attribute == 'No_Beard':
                desc = No_Beard[0]
                res = 'The person probably {}'.format(desc)
            else:
                desc = pretty_print[attribute][0]
                res = 'The person probably {} {}'.format(desc, attribute.replace('_', ' ').lower())
            results_as_strings.append(res)




    # final adjustment
    final_predition = []
    for value in adjusted_prediction:
        if value < uncertainty_threshold:
            final_predition.append(0)
        elif value >= 1 - uncertainty_threshold:
            final_predition.append(1)
        else:
            final_predition.append(value)

    # results - TODO: make cleaner and add if the attribute was adjusted
    results = {
        'sure': {'pos': [], 'neg': []},
        'unsure': {'pos': [], 'neg': []}
    }

    for i, value in enumerate(final_predition):
        name = label_dict[i]
        original_pred = prediction[i]
        adjusted_pred = adjusted_prediction[i] - original_pred
        if value == 0:
            results['sure']['neg'].append({"Attribute": name, "Prediction": "{:.2}".format(original_pred), "Adjusted": "{:.2}".format(adjusted_pred)})
        elif value == 1:
            results['sure']['pos'].append({"Attribute": name, "Prediction": "{:.2}".format(original_pred), "Adjusted": "{:.2}".format(adjusted_pred)})
        elif value < .5:
            results['unsure']['neg'].append({"Attribute": name, "Prediction": "{:.2}".format(original_pred), "Adjusted": "{:.2}".format(adjusted_pred)})
        else:
            results['unsure']['pos'].append({"Attribute": name, "Prediction": "{:.2}".format(original_pred), "Adjusted": "{:.2}".format(adjusted_pred)})

    # results_as_strings = []

    # for item in results['sure']:
    #     if item == 'pos':
    #         index = 0
    #     else:
    #         index = 1
    #     for i in results['sure'][item]:
    #         attribute = i['Attribute']# TODO bearc
    #         if attribute == 'No_Beard':
    #             desc = No_Beard[index]
    #             res = 'The person probably {}'.format(desc)
    #         else:
    #             desc = pretty_print[attribute][index]
    #             res = 'The person probably {} {}'.format(desc, attribute.replace('_', ' ').lower())
    #         results_as_strings.append(res)



    return results, results_as_strings



def md(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)