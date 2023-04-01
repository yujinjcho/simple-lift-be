# from src.process_mp import process_mp
# from src.process_mp2 import process_mp
from src.process_mp3 import process_mp

if __name__ == '__main__':
    print('hello world')

    # input = 'data/deadlift.MOV'
    # output = 'data/processed_deadlift.mp4'

    input = 'data/deadliftsx5.MOV'
    output = 'data/processed_deadliftsx5.mp4'
    process_mp(input, output)
