<script setup>
import {ref, onMounted} from "vue";
import {store} from "../store.js";
import axios from "axios";
import router from "@/router/index.js";
import InfoButton from "@/components/InfoButton.vue";

const isLoading = ref(false);
const loadingText = ref("");
const displayedFrame = ref(null);
const videoId = ref(null)
const frameNum = ref(0)
const totalFrames = ref(0)
const pointType = ref("Additive")
const selectedX = ref(null)
const selectedY = ref(null)
const estimatedX = ref(null)
const estimatedY = ref(null)
const dotX = ref("0px")
const dotY = ref("0px")
const dotSize = ref(40)
const maskedImage = ref(false)
const segmentedVideo = ref(null)

const props = defineProps(['modelValue'])
const emit = defineEmits(['update:modelValue'])
const nextPage = () => {
  emit('update:modelValue', props.modelValue + 1)
}

let cachedFrame = []

onMounted(async () => {
  isLoading.value = true;
  loadingText.value = "Preprocessing video...";
  try {
    videoId.value = store.selectedVideoId;
    if (videoId.value == null) {
      router.push({path: '/'})
      return;
    }
    await axios.get(`${store.apiUrl}/initialize-segmentation?video_id=` + videoId.value)
    loadingText.value = "Getting frame...";
    const response = await axios.get(`${store.apiUrl}/total-frame-count?video_id=` + videoId.value, {
      responseType: "json"
    })

    cachedFrame = new Array(response.data).fill(null);
    frameNum.value = response.data - 1
    totalFrames.value = response.data - 1
    await loadFrame()
  } catch (e) {
    console.error("Failed to load first frame: ", e)
  }
  isLoading.value = false;
  loadingText.value = "";
})

const handleImageClick = (event) => {
  const calcX = Math.floor(event.target.naturalWidth * (event.layerX / event.target.width))
  const calcY = Math.floor(event.target.naturalHeight * (event.layerY / event.target.height))
  estimatedX.value = Math.max(0, calcX);
  estimatedY.value = Math.max(0, calcY);

  dotX.value = Math.floor(event.layerX - dotSize.value/2) + "px"
  dotY.value = Math.floor(event.layerY - dotSize.value/2) + "px"
  selectedY.value = estimatedY
  selectedX.value = estimatedX
  console.log("X: ", calcX, "Y: ", calcY)
}

const loadFrame = async (value) => {
  if (cachedFrame[value] != null) {
    displayedFrame.value = cachedFrame[value];
    return
  }
  const first_frame = await axios.get(`${store.apiUrl}/get-frame?video_id=` + videoId.value + '&frame_num=' + (frameNum.value), {
    responseType: 'blob'
  });
  displayedFrame.value = URL.createObjectURL(first_frame.data);
  cachedFrame[value] = displayedFrame.value
  store.segmentedFrame = displayedFrame.value;
}

const handleDotSubmit = async () => {
  selectedX.value = null
  selectedY.value = null
  isLoading.value = true
  loadingText.value = "Segmenting frame...";
  try {
    const pointFormData = new FormData();
    pointFormData.append('video_id', videoId.value);
    pointFormData.append('point_x', estimatedX.value);
    pointFormData.append('point_y', estimatedY.value);
    pointFormData.append('point_type', pointType.value === "Additive" ? 1 : 0);
    pointFormData.append('frame_num', frameNum.value);
    // Make a POST request to the backend API to apply the blur effect
    const frame_response = await axios.post(`${store.apiUrl}/add-point`, pointFormData, {
      responseType: 'blob'
    });
    console.log(frame_response)
    console.log(frame_response.data)
    displayedFrame.value = URL.createObjectURL(frame_response.data);
    store.segmentedFrame = displayedFrame.value
    maskedImage.value = true;
  } catch (e) {
    console.error(e)
    console.error(e.response.data)
  } finally {
    isLoading.value = false
    loadingText.value = "";
  }
}

const moveToSegmentationResult = async () => {
  isLoading.value = true
  loadingText.value = "Segmenting video...";
  try {
    const resultVideo = await axios.get(`${store.apiUrl}/get-segmentation-result?video_id=` + videoId.value, {
      responseType: 'blob'
    });
    segmentedVideo.value = URL.createObjectURL(resultVideo.data);
    console.log(resultVideo)
    console.log(segmentedVideo.value)
  } catch (e) {
    console.error("Failed to load result: ", e)
  }
  isLoading.value = false
  loadingText.value = "";
}

</script>

<template>
  <v-card elevation="2" class="pa-4 segmentation-card-container">
    <InfoButton>
      <p>Click on the object that you want to apply an effect to. Then click on ‘set point’.
         If you have made a mistake, you can correct it by changing the point type.
         As soon as you are sure that you have selected the correct object, click on ‘continue’.
         The object will now be segmented from the video, this may take some time.
         As soon as the segmentation is complete, you can view the video with the segmentation mask.
         If you are satisfied click ‘continue’, if not start again.
      </p>
    </InfoButton>
    <!-- Card title -->
    <v-card-title class="justify-center">
      <h2>Segmentation</h2>
    </v-card-title>
    <div v-if="!segmentedVideo" class="frame-wrapper">
      <div class="wrapper">
        <img v-if="displayedFrame" :src="displayedFrame" @click.stop="handleImageClick" class="segmentation-image" ismap/>
        <img v-if="selectedX && selectedY" :src="pointType === 'Additive' ? 'src/assets/posDot.svg' : 'src/assets/negDot.svg'" class="select-dot" :width="dotSize" :height="dotSize"/>
      </div>
      <div v-if="!segmentedVideo" class="controls">
        <v-slider
          v-model="frameNum"
          show-ticks="always"
          tick-size="5"
          thumb-label
          :max="totalFrames"
          :min="0"
          :step="1"
          class="pr-5"
          @update:modelValue="loadFrame"
        ></v-slider>
        <v-switch v-model="pointType" :label="`PointType: ${pointType}`" false-value="Subtractive" true-value="Additive" class="point-selection" hide-details />
        <v-btn class="submit-button" :disabled="!selectedX && !selectedY" @click="handleDotSubmit">
          Set Point
        </v-btn>
        <v-btn class="submit-button" :disabled="!maskedImage" @click="moveToSegmentationResult">
          Continue
        </v-btn>
      </div>
      <!-- Loading overlay with centered spinner -->
      <div v-if="isLoading" class="loading-overlay">
        <v-progress-circular indeterminate color="primary" size="50"></v-progress-circular>
        <v-label>{{loadingText}}</v-label>
      </div>
    </div>
    <div v-if="segmentedVideo" class="frame-wrapper">
      <video v-if="segmentedVideo" :src="segmentedVideo" controls muted class="video">
      </video>
      <v-btn class="continue-button" @click="nextPage">
        Continue
      </v-btn>
    </div>
  </v-card>
</template>

<style scoped>

.controls {
  display: flex;
  flex-direction: row;
  align-items: center;
  height: 10%;
}

.controls > * {
  margin-right: 1em;
  margin-top: 0;
}

.wrapper {
  position: relative;
  height: 80%;
}

.select-dot {
  position: absolute;
  z-index: 10;
  left: v-bind('dotX');
  top: v-bind('dotY');
}

.continue-button {
  align-self: flex-end;
  width: 33%;
  min-width: 7em;
}

.video {
  height: 100%;
  width: auto;
}

</style>

<style>

.segmentation-container {
  height: 93vh;
  width: 100%;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: space-evenly;
}

.segmentation-card-container {
  max-width: 1400px;
  width: 100%;
  height: 90%;
  padding: 1em;
}

.segmentation-image {
  position: absolute;
  height: 80%;
}

.frame-wrapper {
  height: 85%;
  width: 100%;
  display: flex;
  flex-direction: column;
}

.point-selection {
  margin-top: 1em;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 8px;
}
</style>