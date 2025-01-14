<script setup>
import { ref } from "vue";
import router from "@/router/index.js";
import { store } from "../store.js";
import axios from "axios";

const isLoading = ref(false);  // Boolean to show a loading spinner while the image is being processed
const fileBlob = ref(null);
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

const resetVideo = () => {
    fileBlob.value = null;
    displayedVideo.value = null;
    store.selectedVideo = null;
    store.selectedVideoId = null;
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

    if (file) {
        fileBlob.value = file; // Store the uploaded file as a Blob
        displayedVideo.value = URL.createObjectURL(file); // Display the uploaded video
        store.selectedVideo = displayedVideo;
        store.videoUrls.push(displayedVideo.value); // Store the uploaded video in the global store
        store.segmentedFrame = null;
    }

    isLoading.value = true;
    try {
        const videoFormData = new FormData();
        videoFormData.append('file', fileBlob.value);
        console.log(fileBlob.value);

        // Make a POST request to the backend API to upload the video
        const id_response = await axios.post(`${store.apiUrl}/upload-video`, videoFormData, {
            responseType: 'json',
            maxContentLength: 10000000,
            maxBodyLength: 100000000
        });
        videoId.value = id_response.data;
        console.log(videoId.value);
        store.selectedVideoId = videoId.value;

        await updateVideoDetails()
        endTime.value = Math.round(duration.value * 100) / 100;

    } catch (error) {
        console.error('Failed to get details:', error);
    } finally {
        isLoading.value = false;
    }
};

const updateVideoDetails = async () => {
    try {
        // Make a GET request to fetch the details of the video
        const details_response = await axios.get(`${store.apiUrl}/video-details?video_id=` + videoId.value);

        const information = details_response.data;
        console.log(information);

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
        }
    } catch (error) {
        console.error('Failed to get details:', error);
    }
}

// Trigger the file input dialog when the image field is clicked
const openFileDialog = () => document.querySelector('input[type="file"]').click();

// Move to segmentation page
const moveToSegmentation = () => router.push({ path: 'segmentation' });

// API call to cut the video
const cutVideo = async () => {
    isLoading.value = true;
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
    }
};
</script>

<template>
  <!-- Wrapper div for positioning the loading overlay -->
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
    </div>

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

    <v-btn class="continue-button" :disabled="!framerate" @click="moveToSegmentation">
      Continue
    </v-btn>
  </div>

  <!-- File input field (hidden) -->
  <v-file-input label="Upload a Video" @change="handleVideoUpload" accept="video/*" class="d-none"
                prepend-icon="mdi-upload"></v-file-input>
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

.video-details {
  align-self: start;
}

.continue-button {
  margin-top: 1em;
}
</style>
