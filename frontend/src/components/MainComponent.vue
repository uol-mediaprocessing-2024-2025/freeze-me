<script setup>
import {ref} from "vue";
import VideoPlayer from "@/components/VideoEditing.vue";
import TimelineComponent from "@/components/TimelineComponent.vue";

// State to manage the visibility of the info popup
const showInfo = ref(false);

// Function to toggle the info popup visibility
const toggleInfo = () => {
  showInfo.value = !showInfo.value;
};
</script>

<template>
  <!-- Main container to center the content on the screen -->
  <v-container class="d-flex align-center justify-center main-container">
    <TimelineComponent/>
    <!-- A card to contain the form and images -->
    <v-card elevation="2" class="pa-4 card-container">
      <!-- Info Button and Popup -->
      <div class="info-button-container">
        <v-btn icon @click="toggleInfo" class="info-button">
          <v-icon>mdi-information</v-icon>
        </v-btn>
        <v-card v-if="showInfo" class="info-popup" elevation="2">
          <v-card-text>
            <p>Upload a video to start a new project or click on ‘projects’ to continue your existing projects.
              As soon as you have uploaded a video, you have the option to cut it.
              In any case, make sure that the object that will later be on your final image is visible in the first frame. </p>
          </v-card-text>
        </v-card>
      </div>

      <!-- Card title -->
      <v-card-title class="justify-center">
        <h2>Video Upload</h2>
      </v-card-title>

      <!-- Card content -->
      <VideoPlayer />
      <v-card-text>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<style scoped>
.main-container {
  height: 93vh;
  width: 100%;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: space-evenly;
}

.card-container {
  max-width: 800px;
  width: 100%;
  position: relative;
}

.info-button-container {
  position: absolute;
  top: 16px;
  right: 16px;
}

.info-button {
  color: #ffffff;
  background-color: #1976d2;
}

.info-popup {
  position: absolute;
  top: 48px;
  right: 16px;
  width: 600px;
  padding: 8px;
  z-index: 10;
}
</style>
