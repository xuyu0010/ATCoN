import os
import sys
import os.path as path

original_file = 'ucf101_msda_test.txt'
new_file = 'ucf101_msda_test_cvt.txt'

sport_classes = ['Archery', 'BaseballPitch', 'Basketball', 'Biking', 'Bowling', 'BreastStroke', 'Diving', 'Fencing', \
'FieldHockeyPenalty', 'FloorGymnastics', 'GolfSwing', 'HorseRace', 'Kayaking', 'RockClimbingIndoor', 'RopeClimbing', 'SkateBoarding', \
'Skiing', 'SumoWrestling', 'Surfing', 'TaiChi', 'TennisSwing', 'TrampolineJumping', 'VolleyballSpiking']
print(len(sport_classes))
if not len(sport_classes) == 23:
	raise Exception

new_lines = []

file = open(original_file, 'r')
for line in file:
	# print(line)
	video_class = line.split()[-1].split('/')[0]

	if video_class in sport_classes:
		class_id = sport_classes.index(video_class)
		cvt_line = line.split()[0] + '\t' + str(class_id) + '\t' + line.split()[-1] + '\n'
		new_lines.append(cvt_line)

file.close()

with open(new_file, 'w') as new_file:
	for line in new_lines:
		new_file.write(line)

new_file.close()
