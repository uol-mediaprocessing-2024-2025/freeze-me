<script setup>
import { computed } from "vue";

const props = defineProps({
  src: { type: String, required: true },
  alt: { type: String, default: "" },
  caption: { type: String, default: "" },
  modelValue: { type: String, default: "" }, // für v-model
  rows: { type: Number, default: 5 },
  placeholder: { type: String, default: "" },
  buttonText: { type: String, default: "Absenden" },
  disabled: { type: Boolean, default: false },
});

const emit = defineEmits(["update:modelValue", "submit"]);

const idTextarea = computed(() => `textarea-${Math.random().toString(36).slice(2, 9)}`);

function onInput(e) {
  emit("update:modelValue", e.target.value);
}

function onSubmit() {
  emit("submit");
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
      :disabled="disabled"
      :value="modelValue"
      @input="onInput"
    ></textarea>

    <button class="c-btn" type="button" :disabled="disabled" @click="onSubmit">
      {{ buttonText }}
    </button>
  </div>
</template>

<style scoped>
.c-wrap {
  display: flex;
  flex-direction: column;
  align-items: center;      /* Bild/Text mittig ausrichten */
  gap: 12px;
  padding: 16px;
  max-width: 700px;
  margin: 0 auto;
}

/* Feste Pixelgröße fürs Bild */
.c-img {
  width: 640px;             /* feste Pixelbreite */
  height: 360px;            /* feste Pixelhöhe */
  object-fit: cover;        /* Bild passend zuschneiden */
  display: block;
}

/* kleiner Text unter dem Bild */
.c-caption {
  font-size: 1rem;
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
}
.c-btn:hover { filter: brightness(1.05); }
.c-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

/* Screenreader-only Label */
.c-sr {
  position: absolute !important;
  height: 1px; width: 1px;
  overflow: hidden;
  clip: rect(1px, 1px, 1px, 1px);
  white-space: nowrap;
}
</style>
