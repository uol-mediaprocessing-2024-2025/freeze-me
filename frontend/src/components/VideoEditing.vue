<script setup>
import {onMounted, ref} from "vue";
import {store} from "../store.js";
import axios from "axios";
import InfoButton from "@/components/InfoButton.vue";
import router from "@/router/index.js";

const isLoading = ref(false);  // Boolean to show a loading spinner while the image is being processed
const loadingText = ref("");
const videoId = ref(null);
const displayedVideo = ref(null);

const framerate = ref(0);
const height = ref(0);
const width = ref(0);
const codec = ref(0);
const bitrate = ref(0);
const duration = ref(0);

const startTime = ref(0);  // Start time for cutting video
const endTime = ref(0);    // End time for cutting video

const props = defineProps(['modelValue'])
const emit = defineEmits(['update:modelValue'])
const nextPage = () => {
  emit('update:modelValue', props.modelValue + 1)
}

onMounted(async () => {
  const id = store.selectedVideoId
  if (id != null) {
    videoId.value = id
    isLoading.value = true
    loadingText.value = "Loading video..."
    if (!store.selectedVideo) {
      const video_response = await axios.get(`${store.apiUrl}/get-video?video_id=` + videoId.value, {responseType: 'blob'})
      console.log(video_response)
      displayedVideo.value = URL.createObjectURL(video_response.data)
      store.selectedVideo = displayedVideo.value
    }
    displayedVideo.value = store.selectedVideo
    isLoading.value = false
    loadingText.value = ""
    await updateVideoDetails()
  } else {
    await router.push("/gallery")
  }
})

const updateVideoDetails = async () => {
  try {
    // Make a GET request to fetch the details of the video
    const details_response = await axios.get(`${store.apiUrl}/video-details?video_id=` + videoId.value);

    const information = details_response.data;
    console.log(information);

    store.totalFrames = information.total_frames
    const streamCount = information.format.nb_streams;
    let videoStream;
    for (let i = 0; i < streamCount; i++) {
      const stream = information.streams.at(i);
      if (stream.codec_type === "video") {
        videoStream = stream;
        break;
      }
    }
    if (videoStream) {
      duration.value = Math.round(videoStream.duration * 100) / 100;
      const fpsString = videoStream.r_frame_rate;
      const slashIndex = fpsString.lastIndexOf("/");
      framerate.value = Math.round((Number(fpsString.substring(0, slashIndex)) / Number(fpsString.substring(slashIndex + 1))) * 100) / 100;
      height.value = videoStream.height;
      width.value = videoStream.width;
      bitrate.value = videoStream.bit_rate;
      codec.value = videoStream.codec_name;
      endTime.value = Math.round(duration.value * 100) / 100;
    }
  } catch (error) {
    console.error('Failed to get details:', error);
  }
}

// API call to cut the video
const cutVideo = async () => {
  isLoading.value = true;
  loadingText.value = "Cutting video...";
  try {
    const cutFormData = new FormData();
    cutFormData.append('video_id', videoId.value);
    cutFormData.append('start_time', startTime.value);
    cutFormData.append('end_time', endTime.value);

    const response = await axios.post(`${store.apiUrl}/cut-video`, cutFormData, {
      responseType: 'blob'
    });

    // Handle the response of the video cut operation
    console.log(response);
    // Save the cut video path in store or show it to the user
    displayedVideo.value = URL.createObjectURL(response.data)
    store.selectedVideo = displayedVideo.value

    await updateVideoDetails()
    endTime.value = duration.value;
    startTime.value = 0;

  } catch (error) {
    console.error("Error cutting the video:", error);
  } finally {
    isLoading.value = false;
    loadingText.value = "";
  }
};
</script>

<template>
  <v-card elevation="2" class="pa-4 card-container">
    <InfoButton>
      <p>Upload a video to start a new project or click on ‘projects’ to continue your existing projects.
        As soon as you have uploaded a video, you have the option to cut it.
        In any case, make sure that the object that will later be on your final image is visible in the first frame.
        Video requirements: The effects only work with a static camera. Minor wobbles are okay,
        but can distort the result. The better the quality of the video, the better the quality of the final image.
        Be aware, this can also increase the editing time.</p>
    </InfoButton>
    <!-- Card title -->
    <v-card-title class="justify-center">
      <h2>Video Editing</h2>
    </v-card-title>
    <!-- Wrapper div for positioning the loading overlay -->
    <div class="video-wrapper">
      <v-responsive class="video-container">
        <!-- Display current video (uploaded or after processing) -->
        <video v-if="displayedVideo && !isLoading" :src="displayedVideo" controls muted class="video"></video>
      </v-responsive>

      <!-- Loading overlay with centered spinner -->
      <div v-if="isLoading" class="loading-overlay">
        <v-progress-circular indeterminate color="primary" size="50"></v-progress-circular>
        <v-label>{{ loadingText }}</v-label>
      </div>
      <div class="d-flex flex-row align-self-start">
        <div class="video-details">
          <p v-if="duration">Duration: {{ duration }}s</p>
          <p v-if="framerate">Framerate: {{ framerate }}fps</p>
          <p v-if="height && width">Dimensions: {{ width }} x {{ height }}</p>
          <p v-if="codec">Codec: {{ String(codec).charAt(0).toUpperCase() + String(codec).slice(1) }}</p>
          <p v-if="bitrate">Bitrate: {{ Math.round(bitrate / 1000) }} kBit/s</p>
        </div>

        <!-- Video cutting options (Input fields for start and end time) -->
        <div v-if="displayedVideo && !isLoading" class="cut-container">
          <div>
            <p>Start Time (in seconds):</p>
            <v-text-field
                v-model="startTime"
                type="number"
                :min="0"
                :max="duration"
                :step="0.1"
                label="Start Time"
                required
            />
          </div>
          <div>
            <p>End Time (in seconds):</p>
            <v-text-field
                v-model="endTime"
                type="number"
                :min="0"
                :max="duration"
                :step="0.1"
                label="End Time"
                required
            />
          </div>

          <v-btn class="cut-video-button" @click="cutVideo">
            Cut Video
          </v-btn>
        </div>
      </div>

      <v-btn class="continue-button" :disabled="!framerate" @click="nextPage">
        Continue
      </v-btn>
    </div>
  </v-card>
</template>

<style>
.cut-container {
  display: flex;
  flex-direction: row;
  margin-top: 20px;
  align-items: center;
  align-self: flex-start;
}

.cut-container > div {
  margin-right: 2em;
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

.video-container {
  align-self: center;
  width: 100%;
}

.video-wrapper {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: end;
}

.video-details {
  align-self: start;
  padding-left: 1em;
  padding-right: 2em;
}

.continue-button {
  margin-top: 1em;
}
</style>
