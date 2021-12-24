
data_points = []
with mp_hands.Hands(static_image_mode=True,    max_num_hands=2,    min_detection_confidence=0.5) as hands:
  for fld_name in sub_folders:
      input_dir = Path.cwd()/f"dataset/{fld_name}"
      IMAGE_FILES = list(input_dir.rglob("*.jpg"))  
      for idx, file in enumerate(IMAGE_FILES):
        file = str(file)
        # Read an image, flip it around y-axis for correct handedness output (see
        # above).
        image = cv2.flip(cv2.imread(file), 1)
        # Convert the BGR image to RGB before processing.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            continue
        image_height, image_width, _ = image.shape
        annotated_image = image.copy()
        land_marks = {}
        for hand_type, hand_landmarks in zip(results.multi_handedness , results.multi_hand_landmarks):
            land_marks[hand_type.classification[0].label] = hand_landmarks

        data_points.append(land_marks)
        # Draw hand world landmarks.
        if not results.multi_hand_world_landmarks:
            continue