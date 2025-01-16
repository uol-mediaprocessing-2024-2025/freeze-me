<script setup>
import {onMounted, ref} from "vue";
import {store} from "@/store.js";
import router from "@/router/index.js";
import axios from "axios";

const isLoading = ref(false);
const videoId = ref(null);
const uploadedBackground = ref(null);
const selectedEffect = ref("blur"); // Standardmäßig Blur-Effekt auswählen
const motionBlurPreview = ref("");
const blurStrength = ref(1);
const blurTransparency = ref(1);
const frameSkip = ref(1); // Frame-Abstand für Multiple Instances
const instanceCount = ref(5); // Anzahl der Instanzen für Multiple Instances
const multipleInstancePreview = ref(""); // Vorschau für Multiple Instances
const transparencyMode = ref("uniform"); // Transparenzmodus: uniform oder gradient
const transparencyStrength = ref(0.5); // Transparenzstärke für den Multiple Instances Effekt

onMounted(async () => {
  isLoading.value = true;
  videoId.value = store.selectedVideoId;
  if (videoId.value == null) {
    router.push({path: '/'});
    return;
  }
  if (store.selectedBackground != null) {
    uploadedBackground.value = store.selectedBackground;
  }
  isLoading.value = false;
});

const handleBackgroundUpload = async (event) => {
  if (isLoading.value) {
    return;
  }
  isLoading.value = true;
  const file = event.target.files[0];
  if (file) {
    try {
      const backgroundFormData = new FormData();
      backgroundFormData.append('file', file);
      backgroundFormData.append('video_id', videoId.value);
      console.log("Uploading background for: " + videoId.value);

      const background = await axios.post(`${store.apiUrl}/upload-background`, backgroundFormData, {
          responseType: 'blob',
      });
      uploadedBackground.value = URL.createObjectURL(background.data);
      store.selectedBackground = background.data;
      console.log("Uploaded Background!");
    } catch (e) {
      console.error(e);
    }
  }
  isLoading.value = false;
};

const openFileDialog = () => document.querySelector('input[type="file"]').click();

const generateImage = async () => {
  if (isLoading.value) {
    return;
  }
  isLoading.value = true;

  try {
    const videoIdParam = "video_id=" + videoId.value;
    const strengthParam = "&blur_strength=" + blurStrength.value;
    const transparencyParam = "&blur_transparency=" + blurTransparency.value;
    const skipParam = "&frame_skip=" + frameSkip.value;
    const preview_image = await axios.get(`${store.apiUrl}/get-motion-blur-preview?` + videoIdParam + strengthParam + transparencyParam + skipParam, {
        responseType: 'blob',
    });

    motionBlurPreview.value = URL.createObjectURL(preview_image.data);
    console.log(motionBlurPreview.value);
  } catch (e) {
    console.error(e);
  }

  isLoading.value = false;
};

const applyMultipleInstancesEffect = async () => {
  if (!videoId.value) {
    alert("Bitte wählen Sie ein Video aus.");
    return;
  }

  isLoading.value = true;

  try {
    // API-Aufruf mit den zusätzlichen Parametern für Transparenz
    const response = await axios.get(
      `${store.apiUrl}/effect/multiple-instances/`,
      {
        params: {
          video_id: videoId.value,
          instance_count: instanceCount.value,
          frame_skip: frameSkip.value,
          transparency_mode: transparencyMode.value, // "uniform" oder "gradient"
          transparency_strength: transparencyStrength.value, // Skaliert von 0 bis 1
        },
        responseType: "blob",
      }
    );

    multipleInstancePreview.value = URL.createObjectURL(response.data);
  } catch (error) {
    console.error("Fehler beim Anwenden des Multiple Instances Effekts:", error);
    alert("Ein Fehler ist aufgetreten. Bitte versuchen Sie es erneut.");
  } finally {
    isLoading.value = false;
  }
};


</script>

<template>
  <main>
    <v-container class="d-flex flex-column align-center justify-center segmentation-container">
      <v-card elevation="2" class="pa-4 segmentation-card-container">
        <v-card-title class="justify-center">
          <h2>Effects</h2>
        </v-card-title>

        <v-tabs v-model="selectedEffect" align-tabs="start">
          <v-tab :value="1">Motion Blur</v-tab>
          <v-tab :value="2">Multiple Instances</v-tab>
        </v-tabs>
        <v-tabs-window v-model="selectedEffect" class="tab w-100">
          <v-tabs-window-item :key="1" :value="1" class="h-100">
            <div class="effect-container h-100">
              <div class="user-input h-100">
                <div class="background-container">
                  <h3 class="pb-2">Background</h3>
                  <div v-if="uploadedBackground" @click="openFileDialog" class="background-preview">
                    <img alt="Uploaded background" :src="uploadedBackground">
                  </div>
                  <div v-else @click="openFileDialog" class="p-5 rounded border w-100 h-75 text-center align-content-center">
                    Click to upload a background
                  </div>
                  <v-file-input label="Upload a Background" @change="handleBackgroundUpload" accept="image/*" class="d-none"
                        prepend-icon="mdi-upload"></v-file-input>
                </div>
                <div class="settings-container">
                  <h3 class="pb-2">Settings</h3>
                  <div class="text-caption">Strength of Blur-Effect ({{blurStrength}})</div>
                  <v-slider v-model="blurStrength" show-ticks="always" tick-size="5" thumb-label :max="5" :min="1" :step="1"></v-slider>
                  <div class="text-caption">Transparency of Blur-Effect ({{blurTransparency}})</div>
                  <v-slider v-model="blurTransparency" show-ticks="always" tick-size="5" thumb-label :max="1" :min="0" :step="0.1"></v-slider>
                  <div class="text-caption">Frame Skip [EXPERIMENTAL] ({{frameSkip}})</div>
                  <v-slider v-model="frameSkip" show-ticks="always" tick-size="5" thumb-label :max="20" :min="0" :step="1"></v-slider>
                </div>
              </div>
              <v-card class="image-preview-container">
                <div>
                  <h3 class="pb-2">Image Preview</h3>
                  <img v-if="motionBlurPreview" :src="motionBlurPreview" alt="preview of generated image" class="image-preview">
                  <p class="pt-5" v-else> Press "Generate Image" to see a preview of the image</p>
                </div>
                <v-btn @click="generateImage" :disabled="isLoading" >Generate Image</v-btn>
              </v-card>
            </div>
          </v-tabs-window-item>

          <v-tabs-window-item :key="2" :value="2">
            <v-container fluid>
              <div class="effect-container">
                <div class="user-input">
                  <h3 class="pb-2">Settings</h3>
                  <div class="text-caption">Number of Instances ({{instanceCount}})</div>
                  <v-slider
                    v-model="instanceCount"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="100"
                    :min="1"
                    :step="1"
                  ></v-slider>
                  <div class="text-caption">Frame Skip ({{frameSkip}})</div>
                  <v-slider
                    v-model="frameSkip"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="200"
                    :min="1"
                    :step="1"
                  ></v-slider>
                  <div class="text-caption">Transparency Mode ({{transparencyMode}})</div>
                  <v-select
                    v-model="transparencyMode"
                    :items="[
                      { text: 'Uniform', value: 'uniform' },
                      { text: 'Gradient', value: 'gradient' },
                    ]"
                    item-text="text"
                    item-value="value"
                    label="Select Transparency Mode"
                  ></v-select>


                  <div class="text-caption">Transparency Strength ({{transparencyStrength}})</div>
                  <v-slider
                    v-model="transparencyStrength"
                    show-ticks="always"
                    tick-size="5"
                    thumb-label
                    :max="1"
                    :min="0"
                    :step="0.01"
                  ></v-slider>
                </div>
                <v-card class="image-preview-container">
                  <div>
                    <h3 class="pb-2">Image Preview</h3>
                    <img
                      v-if="multipleInstancePreview"
                      :src="multipleInstancePreview"
                      alt="preview of generated image"
                      class="image-preview"
                    />
                    <p class="pt-5" v-else>
                      Press "Generate Image" to see a preview of the image
                    </p>
                  </div>
                  <v-btn @click="applyMultipleInstancesEffect" :disabled="isLoading">Generate Image</v-btn>
                </v-card>
              </div>
            </v-container>
          </v-tabs-window-item>
        </v-tabs-window>
      </v-card>
    </v-container>
  </main>
</template>


<style scoped>
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

.background-container {
  width: 100%;
  padding: 1em;
  height: 30%;
}
.background-preview {
  height: 80%;
  width: 100%;
}
.background-preview img{
  max-height: 100%;
  max-width: 100%;
}

.settings-container {
  display: flex;
  flex-direction: column;
  padding: 1em;
  width: 100%;
  height: 70%;
}

.image-preview-container {
  display: flex;
  padding: 1em;
  width: 60%;
  flex-direction: column;
  justify-content: space-between;
}

.image-preview {
  height: 80%;
  width: fit-content;
  max-width: 100%;
  max-height: 80%;
}
</style>
