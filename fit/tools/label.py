HumanAct12 = {
    "A0101": "warm_up_wristankle",
    "A0102": "warm_up_pectoral",
    "A0103": "warm_up_eblowback",
    "A0104": "warm_up_bodylean_right_arm",
    "A0105": "warm_up_bodylean_left_arm",
    "A0106": "warm_up_bow_right",
    "A0107": "warm_up_bow_left",
    "A0201": "walk",
    "A0301": "run",
    "A0401": "jump_handsup",
    "A0402": "jump_vertical",
    "A0501": "drink_bottle_righthand",
    "A0502": "drink_bottle_lefthand",
    "A0503": "drink_cup_righthand",
    "A0504": "drink_cup_lefthand",
    "A0505": "drink_both_hands",
    "A0601": "lift_dumbbell with _right hand",
    "A0602": "lift_dumbbell with _left hand",
    "A0603": "Lift dumbells with both hands",
    "A0604": "lift_dumbbell over head",
    "A0605": "lift_dumbells with both hands and bend legs",
    "A0701": "sit",
    "A0801": "eat_finger_right",
    "A0802": "eat_pie/hamburger",
    "A0803": "Eat with left hand",
    "A0901": "Turn steering wheel",
    "A1001": "Take out phone, call and put phone back",
    "A1002": "Call with left hand",
    "A1101": "boxing_left_right",
    "A1102": "boxing_left_upwards",
    "A1103": "boxing_right_upwards",
    "A1104": "boxing_right_left",
    "A1201": "throw_right_hand",
    "A1202": "throw_both_hands"
}

UTD_MHAD = {
    "1": "  right arm swipe to the left(swipt_left)",
    "2": "  right arm swipe to the right(swipt_right)",
    "3": "  right hand wave(wave)",
    "4": "  two hand front clap(clap)",
    "5": "  right arm throw(throw)",
    "6": "  cross arms in the chest(arm_cross)",
    "7": "  basketball shooting(basketball_shoot)",
    "8": "  draw x(draw_x)",
    "9": "  draw circle(clockwise)(draw_circle_CW)",
    "10": " draw circle(counter clockwise)(draw_circle_CCW)",
    "11": " draw triangle(draw_triangle)",
    "12": " bowling(right hand)(bowling)",
    "13": " front boxing(boxing)",
    "14": " baseball swing from right(baseball_swing)",
    "15": " tennis forehand swing(tennis_swing)",
    "16": " arm curl(two arms)(arm_curl)",
    "17": " tennis serve(tennis_serve)",
    "18": " two hand push(push)",
    "19": " knock on door(knock)",
    "20": " hand catch(catch)",
    "21": " pick up and throw(pickup_throw)",
    "22": " jogging(jog)",
    "23": " walking(walk)",
    "24": " sit to stand(sit2stand)",
    "25": " stand to sit(stand2sit)",
    "26": " forward lunge(left foot forward)(lunge)",
    "27": " squat(squat)"
}


def get_label(file_name, dataset_name):
    if dataset_name == 'HumanAct12':
        key = file_name[-5:]
        return HumanAct12[key]
    elif dataset_name == 'UTD_MHAD':
        key = file_name.split('_')[0][1:]
        return UTD_MHAD[key]