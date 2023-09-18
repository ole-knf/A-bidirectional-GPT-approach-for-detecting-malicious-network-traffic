import ids

alarm_threshold = 0

eval_score = ids.evaluate_scores(0.0005409077857621014, 0)
print(eval_score)
alarm = eval_score <= alarm_threshold
print(alarm)