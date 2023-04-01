from src.process_confidence_intervals import process_confidence
from src.process_rep_count import process_rep_count

if __name__ == '__main__':
    print('starting main')

    input = 'data/deadlift.MOV'
    output = 'data/processed_deadlift.mp4'

    # input = 'data/deadliftsx5.MOV'
    # output = 'data/processed_deadliftsx5.mp4'

    # input = 'data/dl-ending.MOV'
    # output = 'data/dl-ending-processed.mp4'

    # process_confidence(input, output)
    process_rep_count(input, output)
