from src.process_mp import process_mp

if __name__ == '__main__':
    print('hello world')

    input = 'data/deadlift.MOV'
    output = 'data/processed_deadlift.mp4'
    process_mp(input, output)
