<script setup>
import {ref} from "vue";

// State to manage the visibility of the info popup
const showInfo = ref(false);

// Function to toggle the info popup visibility
const toggleInfo = () => {
  showInfo.value = !showInfo.value;
};
</script>

<template>
  <div class="info-button-container">
    <v-btn icon @click="toggleInfo" class="info-button">
      <v-icon>mdi-information</v-icon>
    </v-btn>
    <v-card v-if="showInfo" class="info-popup" elevation="2">
      <v-card-text>
        <slot/>
      </v-card-text>
    </v-card>
  </div>

</template>

<style>

.info-button-container {
  position: absolute;
  top: 16px;
  right: 16px;
  display: flex;
  z-index: 10;
  flex-direction: column;
  justify-content: flex-end;
  align-items: center;
}

.info-button {
  color: #ffffff !important;
  background-color: #1976d2 !important;
  z-index: 15;
  margin-left: auto;
}

.info-popup {
  position: absolute;
  top: calc(100% + 8px);
  right: 0;
  width: 600px;
  padding: 16px;
  z-index: 30;
  background-color: white;
  border-radius: 8px;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
}
</style>