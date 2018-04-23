import csv
import skcriteria.madm.closeness as cl


#
# By : Antony Gozes
#

# install skcriteria and beautifultable via pip
# pip install -U scikit-criteria
# pip install beautifultable

def main():
    #
    # Read the data
    #
    matrix = []
    names = []
    properties = []
    read_data('data_with_missing_values.csv', matrix, names, properties)

    #
    # complete missing data
    #
    matrix = complete_data(matrix)

    #
    # Prepare the weights
    #

    # properties = ['ScreenSize', 'PrimaryCamera', 'SecondaryCamera', 'RAM', 'Battery', 'Memory', 'SDSlot', 'TalkTime',
    #             'Price', 'Announced', 'VoiceControl', 'SoundSpeaker', 'Weight', 'PhysicalKeyboard']

    regular_users_weights = [0.07, 0.07, 0.05, 0.09, 0.11, 0.08, 0.06, 0.09, 0.12, 0.02, 0.05, 0.07, 0.08,
                             0.04]  # sum(weights) = 1
    children_weights = [0.06, 0.07, 0.05, 0.09, 0.09, 0.07, 0.04, 0.04, 0.25, 0.02, 0.06, 0.04, 0.09,
                        0.03]  # sum(weights) = 1
    photographers_weights = [0.1, 0.15, 0.12, 0.09, 0.06, 0.09, 0.07, 0.04, 0.09, 0.02, 0.02, 0.04, 0.07,
                             0.04]  # sum(weights) = 1
    buisness_man_weights = [0.1, 0.07, 0.06, 0.09, 0.02, 0.09, 0.07, 0.11, 0.1, 0.04, 0.09, 0.05, 0.02,
                            0.09]  # sum(weights) = 1
    travelers_weights = [0.08, 0.12, 0.1, 0.06, 0.11, 0.08, 0.07, 0.06, 0.01, 0.05, 0.07, 0.09, 0.07,
                         0.03]  # sum(weights) = 1

    groups_weights = [regular_users_weights, children_weights, photographers_weights, buisness_man_weights,
                      travelers_weights]

    validate_weight_groups(groups_weights)

    #
    # Calculate the balance for the best result, you can mix from all the groups with the wieght values
    #

    # balance vector between the groups
    balance_vector = [0, 0, 0, 0, 1]
    validate_balance_vector(balance_vector)

    #
    # compute the balanced weights
    #

    weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(len(weights)):
        st = 0
        for j in range(len(balance_vector)):
            st += balance_vector[j] * groups_weights[j][i]
        weights[i] = st

    #
    # criteria
    #

    # criteria for what is good value min or max
    criteria = [1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, -1, 1]  # -1 -> minimum is best, 1 -> maximum is best

    #
    # run TOPSIS
    #

    rc = cl.TOPSIS()
    rcc = rc.decide(matrix, criteria, weights)
    display_matrix = to_result_object(names, properties, rcc, weights, True)
    print(to_table_string(display_matrix))


def norm_weights(weights):
    sum_of_weights = sum(weights)
    return [x / sum_of_weights for x in weights]


def print_2d(m):
    print('\n\n')
    for row in m:
        print(row)


def validate_weight_groups(wg):
    for g in wg:
        if abs(sum(g) - 1) >= 1e-4:
            raise Exception("sum of each criteria group must be equal to 1")
        for p in g:
            if not (0 < p < 1):
                raise Exception("each property weight must be great than 0 and less than 1")


def validate_balance_vector(bv):
    if sum(bv) != 1:
        raise Exception("sum of balance vector must be equal to 1")
    if max(bv) > 1 or min(bv) < 0:
        raise Exception("each weight in balance group should be grate or equal to 0 and less or equal to 1")


def read_data(data_file, dst_matrix, dst_names, dst_properties):
    import numpy as np
    try:
        with open(data_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            line_counter = 0
            for row in reader:
                if line_counter == 0:
                    for w in row[1:]:
                        dst_properties.append(str(w).strip())
                # matrix.append([float(x) if str(x).isnumeric() else x for x in row][1:])
                elif line_counter > 0 and len(row) > 0:
                    dst_names.append(row[0])
                    dst_matrix.append([np.nan if not str(x).strip() else float(x) for x in row[1:]])
                line_counter += 1
    except Exception as e:
        raise e


def to_result_object(names, properties, decision, weight, sort=False):
    if 'mtx' not in dir(decision):
        raise Exception('Error in decision value')
    ts = [['Rank', 'Name'] + properties, ['Weight'] + weight]
    tr = []
    for i in range(len(names)):
        tr.append([decision.rank_[i]] + [names[i].strip()] + decision.mtx[i].tolist())
    if sort:
        tr = sorted(tr, key=lambda r: int(r[0]))
    return ts + tr


def to_table_string(result_object):
    if not result_object or len(result_object) <= 2:
        return ''
    from beautifultable import BeautifulTable
    bt = BeautifulTable(max_width=4096, default_padding=4)
    bt.set_style(bt.STYLE_MARKDOWN)
    bt.numeric_precision = 2
    bt.serialno = True
    bt.column_headers = [result_object[0][0], result_object[0][1]] + [
        result_object[0][i + 1] + ' (' + str(result_object[1][i]) + ')' for i in
        range(1, len(result_object[1]))]
    # bt.column_headers[1] = result_object[0][0]
    for dr in result_object[2:]:
        bt.append_row(dr)
    return bt.get_string(recalculate_width=True)


def complete_data(data):
    from imputer import Imputer
    import numpy as np
    from pandas import DataFrame
    cols = len(data[0])

    # headers[0] = columns names
    # headers[1] = is values continues or discrete
    # headers[2] is the column have missing values
    headers = [[None] * cols, [None] * cols, [None] * cols]
    data_np = np.array(data)
    for i in range(len(data[0])):
        headers[0][i] = 'p' + str(i + 1)
        t = is_continue_type_row(data_np[:, i])
        headers[1][i] = t[0]  # continues
        headers[2][i] = t[1]  # has missing data

    imp = Imputer()
    for i in range(len(headers[0])):
        if headers[2][i]:
            td = DataFrame(data_np, columns=headers[0])
            data_np = imp.knn(td, i, max(4, min(10, int(len(data) / 10) + 1)), False)
            if not headers[1][i]:  # non continues values will be rounded
                np.around(data_np[:, i], out=data_np[:, i])
    return data_np.tolist()


def is_continue_type_row(r):
    import numpy as np
    cont = False
    has_missing = False
    for x in r:
        if x is not None and str(x).strip() != '' and not float(x).is_integer() and not np.isnan(float(x)):
            cont = True
        if str(x).strip() == '' or x == np.nan or np.isnan(x) or np.isnan(float(x)):
            has_missing = True
    return cont, has_missing


if __name__ == '__main__':
    main()
