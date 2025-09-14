<script setup>
import { computed, defineEmits, defineProps } from "vue";

const emit = defineEmits(["update:modelValue", "submit"]);

const props = defineProps([
  "modelValue",
  "rows",
  "src",
  "alt",
  "caption",
  "placeholder",
  "buttonText",
]);

const idTextarea = computed(
  () => `textarea-${Math.random().toString(36).slice(2, 9)}`
);

function onInput(e) {
  emit("update:modelValue", e.target.value);
}

function onSubmit() {
  emit("submit");
  console.log(props.src);
}
</script>

<template>
  <div class="c-wrap">
    <img class="c-img" :src="src" :alt="alt" />

    <p v-if="caption" class="c-caption">{{ caption }}</p>

    <label class="c-sr" :for="idTextarea">Eingabe</label>
    <textarea
      :id="idTextarea"
      class="c-textarea"
      :rows="rows"
      :placeholder="placeholder"
      :value="modelValue"
      @input="onInput"
    ></textarea>

    <button class="c-btn" type="button" @click="onSubmit">
      {{ buttonText }}
    </button>
  </div>
</template>

<style scoped>
.c-wrap {
  display: flex;
  flex-direction: column;
  align-items: center; /* Bild/Text mittig ausrichten */
  gap: 12px;
  padding: 16px;
  max-width: 800px;
  margin: 0 auto;
}

/* Feste Pixelgröße fürs Bild */
.c-img {
  width: 1280px; /* feste Pixelbreite */
  height: 720px; /* feste Pixelhöhe */
  object-fit: cover; /* Bild passend zuschneiden */
  display: block;
}

/* kleiner Text unter dem Bild */
.c-caption {
  font-size: 1.4em;
  line-height: 1.3;
  color: #000;
  text-align: left;
  margin-top: 1em;
  margin-bottom: 0.5em;
}

/* Textarea */
.c-textarea {
  width: 100%;
  box-sizing: border-box;
  padding: 10px 12px;
  border: 1px solid #d0d5dd;
  border-radius: 10px;
  font: inherit;
  resize: vertical;
  outline: none;
  font-size: 1.1em;
}

.c-textarea:focus {
  border-color: #7c93ff;
  box-shadow: 0 0 0 3px rgba(124, 147, 255, 0.25);
}

/* Button unten */
.c-btn {
  width: 100%;
  padding: 12px 16px;
  border: none;
  border-radius: 12px;
  font-weight: 600;
  cursor: pointer;
  background: #1e40af;
  color: #fff;
  transition: filter 120ms ease;
  font-size: 1.2em;
}

.c-btn:hover {
  filter: brightness(1.05);
}

.c-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Screenreader-only Label */
.c-sr {
  position: absolute !important;
  height: 1px;
  width: 1px;
  overflow: hidden;
  clip: rect(1px, 1px, 1px, 1px);
  white-space: nowrap;
}
</style>
