<script setup>
import {onMounted, ref, watch} from "vue";
import {store} from "@/store.js";
import axios from "axios";
import router from "@/router/index.js";

  const segmentation = ref(store.steps.segmentation);
  const main_effect = ref(store.steps.mainEffect);
  const after_effect = ref(store.steps.afterEffect);
  const current_step = ref('')
  const video_id = ref(store.selectedVideoId)

  onMounted(async () => {
    if (store.selectedVideoId != null) {
      video_id.value = store.selectedVideoId
      const video_data = (await axios.get(`${store.apiUrl}/project-data?video_id=` + store.selectedVideoId)).data
      console.log(video_data)
      store.steps.segmentation = video_data.available_steps.includes("segmentation")
      store.steps.mainEffect = video_data.available_steps.includes("main-effect")
      store.steps.afterEffect = video_data.available_steps.includes("after-effect")
      current_step.value = video_data.current_step
      console.log(store)
    }
  })

  watch(
    () => store.selectedVideoId,
    async (newVideoId) => {
      if (newVideoId == null) {
        segmentation.value = false;
        main_effect.value = false;
        after_effect.value = false;
        current_step.value = 'video-editing'
        return;
      }
      video_id.value = newVideoId
      const video_data = (await axios.get(`${store.apiUrl}/project-data?video_id=` + store.selectedVideoId)).data
      current_step.value = video_data.current_step
    }
  )

  const goToStep = (step) => {
    switch (step) {
      case 'segmentation':
        current_step.value = 'segmentation'
        router.push({ path: 'segmentation' })
        break;
      case 'main-effect':
        current_step.value = 'main-effect'
        router.push({ path: 'effect-selection' })
        break;
      case 'after-effect':
        current_step.value = 'after-effect'
        router.push({ path: 'final-effects' })
        break;
      default:
        current_step.value = 'video-editing'
        router.push({ path: '/' })
    }
  }

</script>

<template>
  <v-card v-if="video_id" class="progress-line" elevation="4">
    <v-card class="step-card clickable" elevation="3" @click="goToStep('video-editing')">
      <img :src="'src/assets/workflow/video-cut-available.svg'"
         class="progress-icon" :class="current_step === 'video-editing' ? 'current-step' : '' ">
      <p>Video Editing</p>
    </v-card>

    <span class="divider" :class="segmentation ? 'colored-divider' : ''"></span>
    <v-card class="step-card" :class="segmentation ? 'clickable' : ''" elevation="3" @click="goToStep('segmentation')">
      <img :src="segmentation ? 'src/assets/workflow/segmentation-available.svg' : 'src/assets/workflow/segmentation-unavailable.svg'"
         class="progress-icon" :class="current_step === 'segmentation' ? 'current-step' : '' ">
      <p>Segmentation</p>
    </v-card>

    <span class="divider" :class="main_effect ? 'colored-divider' : ''"></span>
    <v-card class="step-card" :class="main_effect ? 'clickable' : ''" elevation="3" @click="goToStep('main-effect')">
      <img :src="main_effect ? 'src/assets/workflow/main-effect-available.svg' : 'src/assets/workflow/main-effect-unavailable.svg'"
         class="progress-icon" :class="current_step === 'main-effect' ? 'current-step' : '' ">
      <p>Main-Effect</p>
    </v-card>

    <span class="divider" :class="after_effect ? 'colored-divider' : ''"></span>
    <v-card class="step-card" :class="after_effect ? 'clickable' : ''" elevation="3" @click="goToStep('after-effect')">
      <img :src="after_effect ? 'src/assets/workflow/after-effect-available.svg' : 'src/assets/workflow/after-effect-unavailable.svg'"
         class="progress-icon" :class="current_step === 'after-effect' ? 'current-step' : '' ">
      <p>After-Effects</p>
    </v-card>
  </v-card>
</template>

<style scoped>
.progress-line {
  display: flex;
  flex-direction: column;
  flex-wrap: nowrap;
  align-items: center;
  padding: 1em 0.5em;
  width: 10em;
  min-width: 8em;
  margin-right: 1em;
}

.step-card {
  justify-items: center;
  width: 7em;
  padding: 0.5em 0;
}

.clickable:hover {
  background: rgba(0,0,0,0.1);
  cursor: pointer;
}

.progress-icon {
  height: 2em;
  width: 2em;
}

.divider {
  height: 2em;
  border-left: 2px solid dimgray;
  margin: 5px
}

.colored-divider {
  border-color: blue;
}

.current-step {
  border: blue 1px solid;
  border-radius: 0.3em;
}
</style>