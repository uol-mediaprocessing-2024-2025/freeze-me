<script setup>
import { onMounted, ref } from 'vue'; // Import lifecycle hook and ref from Vue
import { store } from '../store';// Import the global store to access shared state
import { useRouter } from 'vue-router'; // Import Vue Router for navigation

// Reactive reference to hold the list of images
const videos = ref([]);
const router = useRouter(); // Initialize the router

// onMounted lifecycle hook to fetch images when the component is mounted
onMounted(() => {
    videos.value = store.videoUrls;
});

// Function to handle image click and navigate to the main view
const handleVideoClick = (videoSrc) => {
    store.selectedVideo = videoSrc; // Set the selected image in the store
    router.push('/'); // Navigate to the main view
};
</script>

<template>
  <!-- Gallery container -->
  <div class="gallery px-4 py-4">
      <!-- Vuetify grid to organize images -->
    <v-row dense>
      <v-col v-for="(vidSrc, index) in videos" :key="index" class="d-flex child-flex" cols="12" sm="8" md="6" lg="4" xl="4">
        <div class="rounded border pa-5 video-container">
          <video class="video" @click="handleVideoClick(vidSrc)">
            <source :src="vidSrc" type="video/mp4">
          </video>
        </div>
      </v-col>
    </v-row>
  </div>
</template>

<style scoped>
.gallery {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}

.video-container {
  height: auto;
  width: 400px;
  cursor: pointer;
  transition: transform 0.3s ease-in-out;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  display: flex;
}

.video-container:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
}
</style>
