def write_csv(filename, predictions):
    with open(filename, 'w') as f:
        f.write('Id,Prediction\n')
        for index, prediction in enumerate(predictions):
            f.write(str(index + 1))
            f.write(',')
            f.write(str(prediction))
            f.write('\n')
