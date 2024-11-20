<script setup>
import {ref, watch} from "vue";
import {store} from "../store.js";
import axios from "axios";

const isLoading = ref(false);  // Boolean to show a loading spinner while the image is being processed
const fileBlob = ref(null);
const displayedVideo = ref(null);
const framerate = ref(0)
const height = ref(0)
const width = ref(0)
const codec = ref(0)
const bitrate = ref(0)
const duration = ref(0)

// Watch for changes in the selected image from the gallery
watch(
    () => store.selectedVideo,
    async (newImage) => {
      if (newImage) {
        const response = await fetch(newImage);
        fileBlob.value = await response.blob();  // Convert gallery image to blob
        displayedVideo.value = newImage; // Display the selected image
      }
    },
    {immediate: true}
);

const resetVideo = () => {
  fileBlob.value = null;
  displayedVideo.value = null;
  store.selectedVideo = null;
  framerate.value = 0;
  height.value = 0;
  width.value = 0;
  codec.value = 0;
  bitrate.value = 0;
  duration.value = 0;
  document.querySelector('input[type="file"]').value = '';
};

// Handle image upload
const handleImageUpload = async (event) => {
  const file = event.target.files[0];

  if (file) {
    fileBlob.value = file; // Store the uploaded file as a Blob
    displayedVideo.value = URL.createObjectURL(file); // Display the uploaded video
    store.selectedVideo = displayedVideo;
    store.videoUrls.push(displayedVideo.value); // Store the uploaded image in the global store
  }

  isLoading.value = true;
  try {
    const formData = new FormData();
    formData.append('file', fileBlob.value);
    // Make a POST request to the backend API to apply the blur effect
    const response = await axios.post(`${store.apiUrl}/video-details`, formData, {
      responseType: 'json',
      maxContentLength: 10000000,
      maxBodyLength: 100000000
    });

    const information = response.data
    console.log(information)

    const streamCount = information.format.nb_streams;
    let videoStream;
    for (let i = 0; i < streamCount; i++) {
      const stream = information.streams.at(i)
      if (stream.codec_type === "video") {
        videoStream = stream;
        break;
      }
    }
    if (videoStream) {
      duration.value = videoStream.duration;
      framerate.value = videoStream.r_frame_rate;
      height.value = videoStream.height;
      width.value = videoStream.width;
      bitrate.value = videoStream.bit_rate;
      codec.value = videoStream.codec_name;
    }

  } catch (error) {
    console.error('Failed to get details:', error);
  } finally {
    isLoading.value = false;
  }
};

// Trigger the file input dialog when the image field is clicked
const openFileDialog = () => document.querySelector('input[type="file"]').click();

</script>

<template>
  <!-- Wrapper div for positioning the loading overlay -->
  <div class="video-wrapper">
    <!-- "X" button to reset the image -->
    <v-btn v-if="displayedVideo" icon density="compact" class="reset-btn ma-2" @click="resetVideo" color="red">
      <v-icon small>mdi-close</v-icon>
    </v-btn>
    <v-responsive class="video-container">
      <!-- Display current image (original or blurred) or placeholder -->
      <video v-if="displayedVideo && !isLoading" :src="displayedVideo" controls muted class="video"></video>
      <div @click="openFileDialog" class="d-flex align-center justify-center pa-16 rounded border" v-else>
        Click to upload an image
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
  </div>
  <!-- File input field (hidden) -->
  <v-file-input label="Upload an Image" @change="handleImageUpload" accept="video/*" class="d-none"
                prepend-icon="mdi-upload"></v-file-input>
</template>

<style>
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
</style>