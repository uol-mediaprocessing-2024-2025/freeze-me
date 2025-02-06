<script setup>
import VideoEditing from "@/components/VideoEditing.vue";
import TimelineComponent from "@/components/TimelineComponent.vue";
import EffectSelection from "@/components/EffectSelection.vue";
import VideoUpload from "@/components/VideoUpload.vue";
import FinalEffects from "@/components/FinalEffects.vue";
import SegmentationComponent from "@/components/SegmentationComponent.vue";
import {onMounted, ref} from "vue";
import {store} from "@/store.js";

const allSteps = ref(['video-upload', 'video-editing', 'segmentation', 'main-effects', 'after-effects'])
const currentIndex = ref(0);
const updatePage = (index) => {
  currentIndex.value = index
  console.log(index)
  switch (index) {
    case 2:
      store.steps.segmentation = true;
      break;
    case 3:
      store.steps.mainEffect = true;
      break;
    case 4:
      store.steps.afterEffect = true;
      break;
    default:
      store.steps.videoEditing = true;
  }
  console.log(store.steps)
}

onMounted(() => {
  currentIndex.value += store.steps.videoEditing
  currentIndex.value += store.steps.segmentation
  currentIndex.value += store.steps.mainEffect
  currentIndex.value += store.steps.afterEffect
})
</script>

<template>
  <main>
    <!-- Main container to center the content on the screen -->
    <v-container class="d-flex align-center justify-center main-container main-view">
      <TimelineComponent v-if="currentIndex > 0" :model-value="currentIndex" @update:modelValue="updatePage"/>
      <VideoUpload v-if="allSteps[currentIndex] === allSteps[0]" :model-value="currentIndex" @update:modelValue="updatePage" />
      <VideoEditing v-if="allSteps[currentIndex] === allSteps[1]" :model-value="currentIndex" @update:modelValue="updatePage" />
      <SegmentationComponent v-if="allSteps[currentIndex] === allSteps[2]" :model-value="currentIndex" @update:modelValue="updatePage" />
      <EffectSelection v-if="allSteps[currentIndex] === allSteps[3]" :model-value="currentIndex" @update:modelValue="updatePage" />
      <FinalEffects v-if="allSteps[currentIndex] === allSteps[4]" :model-value="currentIndex" @update:modelValue="updatePage" />
    </v-container>
  </main>
</template>

<style>
.main-view {
  height: 78vh;
  width: 100%;
}

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
</style>
