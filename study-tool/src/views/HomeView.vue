<script setup>
import router from "@/router";
import { store } from "@/store";
import axios from "axios";

const moveToDemographicQuestions = async () => {
  try {
    const answer = await axios.post(`${store.apiUrl}/create_user_folder`, {
      responseType: "json",
    });
    if (answer.status === 200) {
      console.log(answer);
      store.user_path = answer.data;
      router.replace("/demographics");
    }
  } catch (e) {
    console.error(e);
  }
};
</script>

<template>
  <main>
    <h1>Bachelor-Studie</h1>
    <h3>
      Forschungsfrage: Welche Bewegungseffekte in Vorschaubildern können
      einem/einer Nutzer:in helfen, den Inhalt eines Videos zu beschreiben?
    </h3>
    <p>
      Herzlich willkommen und vielen Dank für Ihre Teilnahme an dieser Studie.
    </p>
    <p>
      Ziel der Studie ist es zu untersuchen, wie Menschen Informationen aus
      Bildern und Videos wahrnehmen und einschätzen. Die gesamte Bearbeitung
      dauert etwa 30 Minuten.
    </p>
    <p>
      Ihre Angaben werden anonym gespeichert. Es werden keine personenbezogenen
      Daten erhoben, lediglich einige wenige demografische Angaben.
    </p>
    <p>
      Bitte nehmen Sie sich Zeit und arbeiten Sie die Aufgaben konzentriert
      durch.
    </p>
    <p>
      Wenn Sie bereit sind und keine Fragen mehr haben, können Sie auf den
      Start-Knopf klicken.
    </p>
    <button class="start-button" @click="moveToDemographicQuestions">
      START
    </button>
  </main>
</template>

<style scoped>
main {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.start-button {
  padding: 1em 2em;
  background-color: #1e40af;
  color: #fff;
  border-radius: 0.5em;
  margin: 1em;
  font-size: 1.4em;
  cursor: pointer;
}

.start-button:hover {
  background-color: #0e207f;
}

h3 {
  margin: 1em;
}
</style>
