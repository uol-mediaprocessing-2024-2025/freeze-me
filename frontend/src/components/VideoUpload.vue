<script setup>
import {onMounted, ref} from "vue";
import { store } from "../store.js";
import axios from "axios";
import InfoButton from "@/components/InfoButton.vue";

const isLoading = ref(false);  // Boolean to show a loading spinner while the image is being processed
const loadingText = ref("");  // Boolean to show a loading spinner while the image is being processed
const videoId = ref(null);
const displayedVideo = ref(null);

const framerate = ref(0);
const height = ref(0);
const width = ref(0);
const codec = ref(0);
const bitrate = ref(0);
const duration = ref(0);

const props = defineProps(['modelValue'])
const emit = defineEmits(['update:modelValue'])
const nextPage = () => {
  emit('update:modelValue', props.modelValue + 1)
}

const resetVideo = () => {
    displayedVideo.value = null;
    store.selectedVideo = null;
    store.selectedVideoId = null;
    store.steps.videoEditing = false;
    store.steps.segmentation = false;
    store.steps.mainEffect = false;
    store.steps.afterEffect = false;
    framerate.value = 0;
    height.value = 0;
    width.value = 0;
    codec.value = 0;
    bitrate.value = 0;
    duration.value = 0;
    document.querySelector('input[type="file"]').value = '';
};

// Handle video upload
const handleVideoUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    const fileBlob = file; // Store the uploaded file as a Blob
    displayedVideo.value = URL.createObjectURL(file); // Display the uploaded video
    store.selectedVideo = displayedVideo;

    isLoading.value = true;
    loadingText.value = "Uploading and processing video...";
    try {
        const videoFormData = new FormData();
        videoFormData.append('file', fileBlob);

        // Make a POST request to the backend API to upload the video
        const id_response = await axios.post(`${store.apiUrl}/upload-video`, videoFormData, {
            responseType: 'json',
            maxContentLength: 10000000,
            maxBodyLength: 100000000
        });
        videoId.value = id_response.data;
        console.log(videoId.value);
        store.selectedVideoId = videoId.value;
    } catch (error) {
        console.error('Failed to get details:', error);
    } finally {
        isLoading.value = false;
        loadingText.value = "";
    }
};

// Trigger the file input dialog when the image field is clicked
const openFileDialog = () => document.querySelector('input[type="file"]').click();

</script>

<template>
  <!-- A card to contain the form and images -->
  <v-card elevation="2" class="pa-4 card-container">
    <!-- Info Button and Popup -->
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
      <h2>Video Upload</h2>
    </v-card-title>

        <!-- Card content --><!-- Wrapper div for positioning the loading overlay -->
    <div class="video-wrapper">
      <!-- "X" button to reset the video -->
      <v-btn v-if="displayedVideo" icon density="compact" class="reset-btn ma-2" @click="resetVideo" color="red">
        <v-icon small>mdi-close</v-icon>
      </v-btn>

      <v-responsive class="video-container">
        <!-- Display current video (uploaded or after processing) -->
        <video v-if="displayedVideo && !isLoading" :src="displayedVideo" controls muted class="video"></video>
        <div @click="openFileDialog" class="d-flex align-center justify-center pa-16 rounded border" v-else>
          Click to upload a video
        </div>
      </v-responsive>

      <!-- Loading overlay with centered spinner -->
      <div v-if="isLoading" class="loading-overlay">
        <v-progress-circular indeterminate color="primary" size="50"></v-progress-circular>
        <v-label>{{loadingText}}</v-label>
      </div>

      <v-btn class="continue-button" :disabled="!displayedVideo" @click="nextPage">
        Create Project
      </v-btn>
    </div>

    <!-- File input field (hidden) -->
    <v-file-input label="Upload a Video" @change="handleVideoUpload" accept="video/*" class="d-none"
                  prepend-icon="mdi-upload"></v-file-input>
    <v-card-text>
    </v-card-text>
  </v-card>

</template>

<style>
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

.reset-btn {
  position: absolute;
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

.continue-button {
  margin-top: 1em;
}
</style>
