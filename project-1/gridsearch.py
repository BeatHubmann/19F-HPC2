import subprocess
p = 1
avg = 3
timings = {}
for gc in range(2, 6):
    for ur in range(1, 6):
        for dr in range(1, 11):
            result = {}
            time = 0
            for _ in range(avg):
                output = subprocess.check_output(
                    ['./heat2d', '-p', str(p), '-v', '-gc', str(gc), '-ur', str(ur), '-dr', str(dr)])
                for row in output.decode('utf-8').split('\n'):
                    if ': ' in row:
                        key, value = row.split(': ')
                        result[key.strip()] = value.strip('s')
                time += float(result['Running Time'])
            time /= avg
            timings['gc= ' + str(gc) + ' ur= ' + str(ur) +
                    ' dr= ' + str(dr)] = time
            print(gc, ur, dr, time)
[print(key , ' : ' , value, 's') for (key, value) in sorted(timings.items() ,  key=lambda x: x[1]  ) ]    