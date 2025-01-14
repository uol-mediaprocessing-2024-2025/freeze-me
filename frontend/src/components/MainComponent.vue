<script setup>
import VideoPlayer from "@/components/VideoPlayer.vue";

</script>

<template>
  <!-- Main container to center the content on the screen -->
  <v-container class="d-flex flex-column align-center justify-center main-container">
    <!-- A card to contain the form and images -->
    <v-card elevation="2" class="pa-4 card-container">
      <!-- Card title -->
      <v-card-title class="justify-center">
        <h2>Video Upload</h2>
      </v-card-title>
      <!-- Card content -->
        <VideoPlayer/>
      <v-card-text>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<style scoped>
.main-container {
  height: 78vh;
}

.card-container {
  max-width: 800px;
  width: 100%;
}
</style>