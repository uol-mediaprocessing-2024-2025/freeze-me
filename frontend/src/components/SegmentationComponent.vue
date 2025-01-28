<script setup>
import {ref, onMounted} from "vue";
import {store} from "../store.js";
import axios from "axios";
import router from "@/router/index.js";
import TimelineComponent from "@/components/TimelineComponent.vue";

const isLoading = ref(false);
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
const dotSize = ref(50)
const maskedImage = ref(false)
const segmentedVideo = ref(null)
const showInfo = ref(false);

onMounted(async () => {
  isLoading.value = true;
  try {
    videoId.value = store.selectedVideoId;
    if (videoId.value == null) {
      router.push({path: '/'})
      return;
    }
    await axios.get(`${store.apiUrl}/initialize-segmentation?video_id=` + videoId.value)
    const response = await axios.get(`${store.apiUrl}/total-frame-count?video_id=` + videoId.value, {
      responseType: "json"
    })

    frameNum.value = response.data
    totalFrames.value = response.data
    if (store.segmentedFrame == null) {
      await loadFrame()
    } else {
      displayedFrame.value = store.segmentedFrame
    }
  } catch (e) {
    console.error("Failed to load first frame: ", e)
  }
  isLoading.value = false;
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

const loadFrame = async () => {
  const first_frame = await axios.get(`${store.apiUrl}/get-frame?video_id=` + videoId.value + '&frame_num=' + (frameNum.value - 1), {
    responseType: 'blob'
  });
  displayedFrame.value = URL.createObjectURL(first_frame.data);
  store.segmentedFrame = displayedFrame.value;
}

const handleDotSubmit = async () => {
  selectedX.value = null
  selectedY.value = null
  isLoading.value = true
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
  }
  isLoading.value = false
}

// Function to toggle the info popup visibility
const toggleInfo = () => {
  showInfo.value = !showInfo.value;
}

const moveToSegmentationResult = async () => {
  isLoading.value = true
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
}

const moveToEffectSelection = () => router.push({ path: 'effect-selection' });

</script>

<template>
  <main>
    <v-container class="d-flex flex-column align-center justify-center segmentation-container">
      <!-- A card to contain the form and images -->
      <TimelineComponent/>
      <v-card elevation="2" class="pa-4 segmentation-card-container">
        <div class="info-button-container">
          <v-btn icon @click="toggleInfo" class="info-button">
            <v-icon>mdi-information</v-icon>
          </v-btn>
          <v-card v-if="showInfo" class="info-popup" elevation="2">
            <v-card-text>
              <p>Click on the object that you want to apply an effect to. Then click on ‘set point’.
                 If you have made a mistake, you can correct it by changing the point type.
                 As soon as you are sure that you have selected the correct object, click on ‘continue’.
                 The object will now be segmented from the video, this may take some time.
                 As soon as the segmentation is complete, you can view the video with the segmentation mask.
                 If you are satisfied click ‘continue’, if not start again.
              </p>
            </v-card-text>
          </v-card>
        </div>
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
              :min="1"
              :step="1"
              class="pr-5"
            ></v-slider>
            <v-btn class="submit-button" @click="loadFrame">
              Change Frame
            </v-btn>
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
          </div>
        </div>
        <div v-if="segmentedVideo" class="frame-wrapper">
          <video v-if="segmentedVideo" :src="segmentedVideo" controls muted class="video">
          </video>
          <v-btn class="continue-button" @click="moveToEffectSelection">
            Continue
          </v-btn>
        </div>
      </v-card>
    </v-container>
  </main>

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

.info-button-container {
  position: absolute;
  top: 16px;
  right: 16px;
}

.info-button {
  color: #ffffff !important;
  background-color: #1976d2 !important;
  z-index: 15;
}

.info-popup {
  position: fixed;
  top: 64px;
  right: 16px;
  width: 600px;
  padding: 16px;
  z-index: 10;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
}


</style>