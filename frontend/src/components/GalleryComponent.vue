<script setup>
import { onMounted, ref } from 'vue'; // Import lifecycle hook and ref from Vue
import { store } from '../store';// Import the global store to access shared state
import { useRouter } from 'vue-router';
import axios from "axios"; // Import Vue Router for navigation

// Reactive reference to hold the list of images
const videos = ref([]);
const router = useRouter(); // Initialize the router

// onMounted lifecycle hook to fetch images when the component is mounted
onMounted(async () => {
  const video_ids = (await axios.get(`${store.apiUrl}/video-ids`)).data;
  console.log(video_ids)
  for (let i = 0; i < video_ids.length; i++) {
    const id = video_ids[i];
    const video_data = (await axios.get(`${store.apiUrl}/project-data?video_id=` + id)).data
    const thumbnail_data = await axios.get(`${store.apiUrl}/get-frame?video_id=` + id + '&frame_num=0', {
          responseType: 'blob'
        })
    let thumbnail = null
    thumbnail = URL.createObjectURL(thumbnail_data.data)
    video_data.thumbnail = thumbnail
    videos.value.push(video_data)
  }
  console.log(videos.value)
});

// Function to handle image click and navigate to the main view
const handleVideoClick = (video) => {
  store.selectedVideoId = video.id; // Set the selected image in the store
  store.steps.videoEditing = video.available_steps.indexOf('video-editing') >= 0
  store.steps.segmentation= video.available_steps.indexOf('segmentation') >= 0
  store.steps.background= video.available_steps.indexOf('background') >= 0
  store.steps.mainEffect= video.available_steps.indexOf('main-effect') >= 0
  store.steps.afterEffect= video.available_steps.indexOf('after-effect') >= 0

  let navigation = store.steps.segmentation ? "/segmentation" : "/"
  navigation = store.steps.background ? "/" : navigation
  navigation = store.steps.mainEffect ? "/effect-selection" : navigation
  navigation = store.steps.afterEffect ? "/" : navigation
  router.push(navigation)
};
</script>

<template>
  <!-- Gallery container -->
  <div class="gallery px-4 py-4">
      <!-- Vuetify grid to organize images -->
    <v-row dense>
      <v-col v-for="(video, index) in videos" :key="index" class="d-flex child-flex" cols="12" sm="8" md="6" lg="4" xl="4">
        <div class="rounded border pa-5 video-container d-flex flex-sm-column">
          <img class="video" :src="video.thumbnail" @click="handleVideoClick(video)" alt="Thumbnail of video">
          <div class="progress-line">
            <img :src="video.available_steps.indexOf('video-editing') >= 0 ? 'src/assets/workflow/video-cut-available.svg' : 'src/assets/workflow/video-cut-unavailable.svg'"
                 class="progress-icon" :class="video.current_step == 'video-editing' ? 'current-step' : '' ">
            <span class="divider"></span>
            <img :src="video.available_steps.indexOf('segmentation') >= 0 ? 'src/assets/workflow/segmentation-available.svg' : 'src/assets/workflow/segmentation-unavailable.svg'"
                 class="progress-icon" :class="video.current_step == 'segmentation' ? 'current-step' : '' ">
            <span class="divider"></span>
            <img :src="video.available_steps.indexOf('background') >= 0 ? 'src/assets/workflow/background-available.svg' : 'src/assets/workflow/background-unavailable.svg'"
                 class="progress-icon" :class="video.current_step == 'background' ? 'current-step' : '' ">
            <span class="divider"></span>
            <img :src="video.available_steps.indexOf('main-effect') >= 0 ? 'src/assets/workflow/main-effect-available.svg' : 'src/assets/workflow/main-effect-unavailable.svg'"
                 class="progress-icon" :class="video.current_step == 'main-effect' ? 'current-step' : '' ">
            <span class="divider"></span>
            <img :src="video.available_steps.indexOf('after-effect') >= 0 ? 'src/assets/workflow/after-effect-available.svg' : 'src/assets/workflow/after-effect-unavailable.svg'"
                 class="progress-icon" :class="video.current_step == 'after-effect' ? 'current-step' : '' ">
          </div>
        </div>
      </v-col>
    </v-row>
  </div>
</template>

<style scoped>

.progress-line {
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  align-items: center;
  padding-top: 0.5em;
}

.progress-icon {
  height: 2em;
  width: 2em;
}

.divider {
  flex-grow: 1;
  border-bottom: 1px solid dimgray;
  margin: 5px
}

.current-step {
  border: blue 1px solid;
  border-radius: 0.3em;
}

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
