import sys
import time

from pyspark import SparkContext
from itertools import combinations


def get_new_candidate_items(frequent_items, size):
    candidate_items = set()
    for comb in combinations(frequent_items, 2):
        comb = comb[0].union(comb[1])
        if size - len(comb) == 0:
            candidate_items.add(frozenset(comb))

    return candidate_items


def a_priori(chunk, support):
    baskets = []
    frequency_map = dict()
    # count occurrences

    # frequent item set of size 1
    frequent_items = []
    for basket in chunk:
        baskets.append(basket)
        for item in basket:
            frequency_map[item] = frequency_map.get(item, 0) + 1
            if frequency_map[item] >= support:
                frequent_items.append(set([item]))

    frequent_itemsets = frequent_items
    size = 2

    while len(frequent_items) != 0:
        candidate_items = get_new_candidate_items(frequent_items, size)
        frequency_map = dict()
        frequent_items = []
        for candidate_item in candidate_items:
            for basket in baskets:
                if set(candidate_item).issubset(basket):
                    frequency_map[candidate_item] = frequency_map.get(candidate_item, 0) + 1

        for k, v in frequency_map.items():
            if v >= support:
                frequent_items.append(k)

        frequent_itemsets += frequent_items
        size += 1
    return frequent_itemsets


def get_candidate_freq(candidates, chunk):
    frequency_map = dict()
    chunk_list = list(chunk)

    for candidate in candidates:
        for basket in chunk_list:
            if set(candidate).issubset(set(basket)):
                frequency_map[candidate] = frequency_map.get(candidate, 0) + 1

    return frequency_map.items()


def get_output_txt(frequent_itemsets):
    phase_2_output = ""
    prev_len = 1
    for item_set in frequent_itemsets:
        curr_len = len(item_set)
        if prev_len != curr_len:
            phase_2_output = phase_2_output[:-1]
            phase_2_output += "\n\n"
        prev_len = curr_len
        if curr_len == 1:
            phase_2_output += "('" + item_set[0] + "'),"
        else:
            phase_2_output += str(item_set) + ','

    phase_2_output = phase_2_output[:-1]
    return phase_2_output


def son(baskets, support, output_filepath):
    num_of_partitions = baskets.getNumPartitions()
    # print('num_of_partitions: ', num_of_partitions)

    # phase 1
    scaled_support = support/num_of_partitions
    phase_1 = baskets\
        .mapPartitions(lambda chunk: a_priori(chunk, scaled_support)) \
        .map(lambda s: (tuple(sorted(s)), 1))\
        .reduceByKey(lambda x, y: y).keys()
    candidates = phase_1.collect()
    candidates = sorted(candidates, key=(lambda s: (len(s), s)))
    # print('phase_1', candidates)

    phase_1_output = get_output_txt(candidates)
    with open(output_filepath, "w+") as f:
        f.write("Candidates:\n")
        f.write(phase_1_output)
        f.write("\n\n")

    # phase 2
    phase_2 = baskets\
        .mapPartitions(lambda chunk: get_candidate_freq(candidates, chunk))\
        .reduceByKey(lambda x, y: x+y)\
        .filter(lambda s: s[1] >= support).keys()
    frequent_itemsets = phase_2.collect()
    frequent_itemsets = sorted(frequent_itemsets, key=(lambda s: (len(s), s)))
    # print('phase_2 ', frequent_itemsets)

    phase_2_output = get_output_txt(frequent_itemsets)
    with open(output_filepath, "a") as f:
        f.write("Frequent Itemsets:\n")
        f.write(phase_2_output)


def execute_task1():
    start_time = time.time()
    if len(sys.argv) > 3:
        case_num = int(sys.argv[1])
        support = int(sys.argv[2])
        input_filepath = sys.argv[3]
        output_filepath = sys.argv[4]
    else:
        case_num = 1
        support = 4
        input_filepath = './small1.csv'
        # input_filepath = './small2.csv'
        output_filepath = './output_task1.json'

    sc = SparkContext('local[*]', 'Task 1')
    # remove first line
    data_rdd = sc.textFile(input_filepath).map(lambda s: s.split(","))
    first_line = data_rdd.first()
    data_rdd = data_rdd.filter(lambda s: s != first_line)

    if case_num == 1:
        data_rdd = data_rdd.map(lambda s: (s[0], s[1]))
    else:
        data_rdd = data_rdd.map(lambda s: (s[1], s[0]))
    # baskets = data_rdd.groupByKey().map(lambda s: list(s[1]))
    baskets = data_rdd.groupByKey().mapValues(set).values().persist()

    son(baskets, support, output_filepath)

    print('Duration: ', time.time() - start_time)


if __name__ == '__main__':
    execute_task1()
