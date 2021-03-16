## Transfer Learning 을 하기 위한 코드와 설명
- Transfer Learning이란 사전 학습 된 모델을 이용하는 것을 말한다. 다음과 같이 사용 할 수 있다.
- base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

### Freezing the base model

- base_model.trainable = False

### Defining the custom head for our network

- base_model.output
- head_model = base_model.output
- head_model = tf.keras.layers.GlobalAveragePooling2D()(head_model)
- head_model
- head_model = tf.keras.layers.Dense(units=1, activation='sigmoid')(head_model)

### Defining the model

- model = tf.keras.models.Model(inputs=base_model.input, outputs=head_model)

### Compiling the model

- model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), loss="binary_crossentropy", metrics=["accuracy"])

### Creating Data Generators

### Resizing images

- data_gen_train = ImageDataGenerator(rescale=1/255.)
- data_gen_valid = ImageDataGenerator(rescale=1/255.)

- train_generator = data_gen_train.flow_from_directory(train_dir, target_size=(128,128), batch_size=128, class_mode="binary")
- valid_generator = data_gen_valid.flow_from_directory(validation_dir, target_size=(128,128), batch_size=128, class_mode="binary")

### Training the model

- model.fit(train_generator, epochs=5, validation_data=valid_generator)

### Transfer learning model evaluation

- valid_loss, valid_accuracy = model.evaluate(valid_generator)

## Fine Tuning을 하기 위한 코드와 설명
- Fine Tuning이란 사전 학습된 Transfer Learning을 이용하여 새로운 모델을 학습하는 과정을 말한다. 사용법 예시는 다음과 같다.
- base_model.trainable = True
- print("Number of layersin the base model: {}".format(len(base_model.layers)))
- fine_tune_at = 100
- for layer in base_model.layers[:fine_tune_at]:
-    layer.trainable = False

### Compiling the model for fine-tuning

- model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
-               loss='binary_crossentropy',
-               metrics=['accuracy'])

### Fine tuning

- history = model.fit(train_generator,  
-                       epochs=5, 
-                       validation_data=valid_generator)
                    
## 에포크 시마다, 가장 좋은 모델을 저장하는 ModelCheckpoint 사용방법
- ModelCheckpoint를 이용하면, 에포크 시마다 val_accuracy를 비교하여 정확도가 더 높은 모델을 저장한다.
- 사용법은, 저장할 디렉토리를 먼저 지정해주고 라이브러리를 임포트하여 사용 할 수 있다.
- from tensorflow.keras.callbacks import ModelCheckpoint

- cp = ModelCheckpoint(filepath=CHECKPOINT_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

## 에포크 시마다, 기록을 남길 수 있는, CSVLogger 사용방법
- 에포크 시마다 에포크 횟수와, loss, accuracy등의 기록을 남길수 있다.
- ModelCheckpoint와 마찬가지로 저장할 디렉토리를 지정하고, 라이브러리 임포트 하여 숑한다.
- from tensorflow.keras.callbacks import CSVLogger

- csv_logger = CSVLogger(filename=LOGFILE_PATH, append=True)


- ModelCheckpoint , CVSLogger 예시
- history = model.fit(trainGen.flow(X_train, y_train, batch_size=64), steps_per_epoch = len(X_train) // 64, epochs=50,
-                     validation_data = (X_val, y_val), validation_steps = len(X_val) // 64, callbacks=[cp, csv_logger])
