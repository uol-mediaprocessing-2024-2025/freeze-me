<script setup>
import { ref, onMounted, watch } from "vue";
import { store } from "@/store.js";
import router from "@/router/index.js";
import axios from "axios";
import TimelineComponent from "@/components/TimelineComponent.vue";

const isLoading = ref(false);
const videoId = ref(null);
const selectedEffect = ref("motion_blur"); // Standardmäßig Motion Blur
const previewUrl = ref(""); // URL des Bildes für die Vorschau
const brightness = ref(100); // Prozentuale Helligkeit
const contrast = ref(100); // Prozentualer Kontrast
const saturation = ref(100); // Prozentuale Sättigung
const showInfo = ref(false);

onMounted(async () => {
  videoId.value = store.selectedVideoId;
  if (!videoId.value) {
    router.push({ path: "/" });
    return;
  }
  await loadPreview(); // Initiales Laden des Bildes
});

const loadPreview = async () => {
  if (isLoading.value) return;
  isLoading.value = true;
  try {
    const response = await axios.get(`${store.apiUrl}/final-effects-preview`, {
      params: { video_id: videoId.value, effect_type: selectedEffect.value },
      responseType: "blob",
    });
    previewUrl.value = URL.createObjectURL(response.data);
  } catch (error) {
    console.error("Error loading preview:", error);
    alert("Failed to load preview image. Please try again.");
  } finally {
    isLoading.value = false;
  }
};

const updatePreview = async () => {
  if (isLoading.value) return;
  isLoading.value = true;
  try {
    const response = await axios.post(
      `${store.apiUrl}/apply-final-effects`,
      {
        video_id: videoId.value,
        brightness: brightness.value / 100,
        contrast: contrast.value / 100,
        saturation: saturation.value / 100,
      },
      { responseType: "blob" }
    );
    previewUrl.value = URL.createObjectURL(response.data);
  } catch (error) {
    console.error("Error updating preview:", error);
    alert("Failed to apply effects. Please try again.");
  } finally {
    isLoading.value = false;
  }
};

watch(selectedEffect, async () => {
  await loadPreview(); // Bild neu laden bei Effektwechsel
});

const downloadImage = () => {
  const link = document.createElement("a");
  link.href = previewUrl.value;
  link.download = `${selectedEffect.value}-edited-image.png`;
  link.click();
};

// Function to toggle the info popup visibility
const toggleInfo = () => {
  showInfo.value = !showInfo.value;
};
</script>

<template>
  <main>
    <v-container class="d-flex align-center justify-center segmentation-container">
      <TimelineComponent/>
      <v-card elevation="2" class="pa-4 segmentation-card-container">
        <div class="info-button-container">
        <v-btn icon @click="toggleInfo" class="info-button">
          <v-icon>mdi-information</v-icon>
        </v-btn>
        <v-card v-if="showInfo" class="info-popup" elevation="2">
          <v-card-text>
            <p>Select which image you want to edit.
               Adjust the brightness, contrast and saturation and download your result. </p>
          </v-card-text>
        </v-card>
        </div>
        <v-card-title class="justify-center">
          <h2>Final Effects</h2>
        </v-card-title>

        <v-tabs v-model="selectedEffect" align-tabs="start">
          <v-tab :value="'motion_blur'">Motion Blur</v-tab>
          <v-tab :value="'multiple_instances'">Multiple Instances</v-tab>
        </v-tabs>
        <v-tabs-window v-model="selectedEffect" class="tab w-100">
          <v-tabs-window-item :key="'motion_blur'" :value="'motion_blur'" class="h-100">
            <div class="effect-container h-100">
              <div class="user-input h-100">
                <div class="settings-container">
                  <h3 class="pb-2">Settings</h3>
                  <div class="text-caption">Brightness ({{ brightness }})</div>
                  <v-slider
                    v-model="brightness"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="0"
                    :step="1"
                    @change="updatePreview"
                  ></v-slider>
                  <div class="text-caption">Contrast ({{ contrast }})</div>
                  <v-slider
                    v-model="contrast"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="0"
                    :step="1"
                    @change="updatePreview"
                  ></v-slider>
                  <div class="text-caption">Saturation ({{ saturation }})</div>
                  <v-slider
                    v-model="saturation"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="0"
                    :step="1"
                    @change="updatePreview"
                  ></v-slider>
                </div>
              </div>
              <v-card class="image-preview-container">
                <div>
                  <h3 class="pb-2">Image Preview</h3>
                  <img
                    v-if="previewUrl"
                    :src="previewUrl"
                    alt="Preview of final image"
                    class="image-preview"
                    :style="{
                      filter: `
                        brightness(${brightness}%)
                        contrast(${contrast}%)
                        saturate(${saturation}%)
                      `,
                    }"
                  />
                  <p v-else class="pt-5">No preview available</p>
                </div>
                <v-btn color="secondary" @click="downloadImage" :disabled="!previewUrl">
                  Download Image
                </v-btn>
              </v-card>
            </div>
          </v-tabs-window-item>

          <v-tabs-window-item :key="'multiple_instances'" :value="'multiple_instances'" class="h-100">
            <div class="effect-container h-100">
              <div class="user-input h-100">
                <div class="settings-container">
                  <h3 class="pb-2">Settings</h3>
                  <div class="text-caption">Brightness ({{ brightness }})</div>
                  <v-slider
                    v-model="brightness"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="0"
                    :step="1"
                    @change="updatePreview"
                  ></v-slider>
                  <div class="text-caption">Contrast ({{ contrast }})</div>
                  <v-slider
                    v-model="contrast"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="0"
                    :step="1"
                    @change="updatePreview"
                  ></v-slider>
                  <div class="text-caption">Saturation ({{ saturation }})</div>
                  <v-slider
                    v-model="saturation"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="0"
                    :step="1"
                    @change="updatePreview"
                  ></v-slider>
                </div>
              </div>
              <v-card class="image-preview-container">
                <div>
                  <h3 class="pb-2">Image Preview</h3>
                  <img
                    v-if="previewUrl"
                    :src="previewUrl"
                    alt="Preview of final image"
                    class="image-preview"
                    :style="{
                      filter: `
                        brightness(${brightness}%)
                        contrast(${contrast}%)
                        saturate(${saturation}%)
                      `,
                    }"
                  />
                  <p v-else class="pt-5">No preview available</p>
                </div>
                <v-btn color="secondary" @click="downloadImage" :disabled="!previewUrl">
                  Download Image
                </v-btn>
              </v-card>
            </div>
          </v-tabs-window-item>
        </v-tabs-window>
      </v-card>
    </v-container>
  </main>
</template>

<style scoped>
.segmentation-container {
  height: 93vh;
  width: 100%;
  display: flex;
  flex-direction: row;
  flex-wrap: nowrap;
  justify-content: space-evenly;
}

.segmentation-card-container {
  max-width: 1400px;
  width: 100%;
  height: 90%;
  padding: 1em;
}

.tab {
  height: 90%;
}

.effect-container {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: space-evenly;
}

.user-input {
  display: flex;
  flex-direction: column;
  flex-wrap: nowrap;
  width: 35%;
}

.settings-container {
  display: flex;
  flex-direction: column;
  padding: 1em;
  width: 100%;
}

.image-preview-container {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  width: 60%;
  padding: 1em;
}

.image-preview {
  height: 80%;
  width: fit-content;
  max-width: 100%;
  max-height: 80%;
  border: 1px solid #ccc;
  border-radius: 8px;
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
