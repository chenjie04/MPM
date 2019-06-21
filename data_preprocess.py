import os
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from collections import namedtuple

from load import implicit_load

from mlperf_compliance import mlperf_log


MIN_RATINGS = 20


USER_COLUMN = 'user_id'
ITEM_COLUMN = 'item_id'


TRAIN_RATINGS_FILENAME = 'train_ratings.csv'
TEST_RATINGS_FILENAME = 'test_ratings.csv'
TEST_NEG_FILENAME = 'test_negative.csv'
DATA_SUMMARY_FILENAME = "data_summary.csv"

PATH = 'data/taobao-1m'
OUTPUT = 'data/taobao-1m'
NEGATIVES = 99
HISTORY_SIZE = 9
RANDOM_SEED = 0

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--file',type=str,default=(os.path.join(PATH,'UserBehavior01.csv')),
                        help='Path to reviews CSV file from dataset')
    parser.add_argument('--output', type=str, default=OUTPUT,
                        help='Output directory for train and test CSV files')
    parser.add_argument('-n', '--negatives', type=int, default=NEGATIVES,
                        help='Number of negative samples for each positive'
                             'test example')
    parser.add_argument('--history_size',type=int,default=HISTORY_SIZE,
                        help='The size of history')
    parser.add_argument('-s', '--seed', type=int, default=RANDOM_SEED,
                        help='Random seed to reproduce same negative samples')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("Loading raw data from {}".format(args.file))
    #-------------- MovieLens dataset ------------------------------
    # df = implicit_load(args.file, sort=False)
    #---------------------------------------------------------------

    #------ retailrocket-recommender-system-dataset --------------------
    # df = pd.read_csv(args.file, sep=',', header=0)
    # df.columns = ['timestamp', 'user_id', 'event', 'item_id', 'transaction_id']
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    #
    #
    # RatingData = namedtuple('RatingData',
    #                         ['items', 'users', 'ratings', 'min_date', 'max_date'])
    # info = RatingData(items=len(df['item_id'].unique()),
    #                   users=len(df['user_id'].unique()),
    #                   ratings=len(df),
    #                   min_date=df['timestamp'].min(),
    #                   max_date=df['timestamp'].max())
    # print("{ratings} ratings on {items} items from {users} users"
    #           " from {min_date} to {max_date}"
    #           .format(**(info._asdict())))
    # #--------------------------------------------------------------------

    #-------------------amazon dataset------------------------
    # df = pd.read_csv(args.file, sep=',', header=None)
    # df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    #
    # RatingData = namedtuple('RatingData',
    #                         ['items', 'users', 'ratings', 'min_date', 'max_date'])
    # info = RatingData(items=len(df['item_id'].unique()),
    #                   users=len(df['user_id'].unique()),
    #                   ratings=len(df),
    #                   min_date=df['timestamp'].min(),
    #                   max_date=df['timestamp'].max())
    # print("{ratings} ratings on {items} items from {users} users"
    #           " from {min_date} to {max_date}"
    #           .format(**(info._asdict())))


    #-------------------------------------------------------------------------

    #------------------- hetrec2011 dataset------------------------
    # df = pd.read_csv(args.file, sep='\t', header=0)
    # df.columns = ['user_id', 'item_id', 'tag_id', 'timestamp']
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    #
    # RatingData = namedtuple('RatingData',
    #                         ['items', 'users', 'ratings', 'min_date', 'max_date'])
    # info = RatingData(items=len(df['item_id'].unique()),
    #                   users=len(df['user_id'].unique()),
    #                   ratings=len(df),
    #                   min_date=df['timestamp'].min(),
    #                   max_date=df['timestamp'].max())
    # print("{ratings} ratings on {items} items from {users} users"
    #           " from {min_date} to {max_date}"
    #           .format(**(info._asdict())))
    #

    #-------------------------------------------------------------------------

    #------------------- taobao UserBehavior dataset------------------------
    df = pd.read_csv(args.file, sep=',', header=None)
    df.columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

    RatingData = namedtuple('RatingData',
                            ['items', 'users', 'ratings', 'min_date', 'max_date'])
    info = RatingData(items=len(df['item_id'].unique()),
                      users=len(df['user_id'].unique()),
                      ratings=len(df),
                      min_date=df['timestamp'].min(),
                      max_date=df['timestamp'].max())
    print("{ratings} ratings on {items} items from {users} users"
              " from {min_date} to {max_date}"
              .format(**(info._asdict())))


    #-------------------------------------------------------------------------

    print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
    grouped = df.groupby(USER_COLUMN)
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_MIN_RATINGS, value=MIN_RATINGS)
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    print("Mapping original user and item IDs to new sequential IDs")
    original_users = df[USER_COLUMN].unique()
    original_items = df[ITEM_COLUMN].unique()

    nb_users = len(original_users)
    nb_items = len(original_items)

    user_map = {user: index for index, user in enumerate(original_users)}
    item_map = {item: index for index, item in enumerate(original_items)}

    df[USER_COLUMN] = df[USER_COLUMN].apply(lambda user: user_map[user])
    df[ITEM_COLUMN] = df[ITEM_COLUMN].apply(lambda item: item_map[item])

    # print(df)


    assert df[USER_COLUMN].max() == len(original_users) - 1
    assert df[ITEM_COLUMN].max() == len(original_items) - 1

    print("Creating list of items for each user")
    # Need to sort before popping to get last item
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.sort_values(by='timestamp', inplace=True)
    all_ratings = set(zip(df[USER_COLUMN], df[ITEM_COLUMN]))
    user_to_items = defaultdict(list)
    for row in tqdm(df.itertuples(), desc='Ratings', total=len(df)):
        user_to_items[getattr(row, USER_COLUMN)].append(getattr(row, ITEM_COLUMN))  # noqa: E501

    print(len(user_to_items[0]))
    print(user_to_items[0])
    print(user_to_items[0][-args.history_size:])



    print("Generating {} negative samples for each user and creating training set"
          .format(args.negatives))
    mlperf_log.ncf_print(key=mlperf_log.PREPROC_HP_NUM_EVAL, value=args.negatives)

    train_ratings = []
    test_ratings = []
    test_negs = []
    all_items = set(range(len(original_items)))

    for key, value in tqdm(user_to_items.items(), total=len(user_to_items)):
        all_negs = all_items - set(value)
        all_negs = sorted(list(all_negs))
        negs = random.sample(all_negs, args.negatives)

        test_item = value.pop()

        tmp = [key, test_item]
        tmp.extend(negs)
        test_negs.append(tmp)

        tmp = [key, test_item]
        tmp.extend(value[-args.history_size:])
        test_ratings.append(tmp)

        while len(value) > args.history_size:
            tgItem = value.pop()
            tmp = [key,tgItem]
            tmp.extend(value[-args.history_size:])
            train_ratings.append(tmp)



    print("\nSaving train and test CSV files to {}".format(args.output))



    df_train_ratings = pd.DataFrame(list(train_ratings))
    df_test_ratings = pd.DataFrame(list(test_ratings))
    df_test_negs = pd.DataFrame(list(test_negs))


    print('Saving data description ...')
    data_summary = pd.DataFrame(
        {'users': nb_users, 'items': nb_items, 'history_size': HISTORY_SIZE, 'train_entries': len(df_train_ratings), 'test': len(df_test_ratings)},
        index=[0])
    data_summary.to_csv(os.path.join(args.output, DATA_SUMMARY_FILENAME), header=True, index=False, sep=',')

    df_train_ratings['fake_rating'] = 1
    df_train_ratings.to_csv(os.path.join(args.output, TRAIN_RATINGS_FILENAME),
                            index=False, header=False, sep='\t')

    mlperf_log.ncf_print(key=mlperf_log.INPUT_SIZE, value=len(df_train_ratings))


    df_test_ratings['fake_rating'] = 1
    df_test_ratings.to_csv(os.path.join(args.output, TEST_RATINGS_FILENAME),
                           index=False, header=False, sep='\t')


    df_test_negs.to_csv(os.path.join(args.output, TEST_NEG_FILENAME),
                        index=False, header=False, sep='\t')




if __name__ == '__main__':
    main()
