package com.samassist.testshell

import android.content.Context
import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts.OpenDocument
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    VisionTestShellScreen()
                }
            }
        }
    }
}


private data class VisionUiResult(
    val rawJson: String,
    val description: String,
    val userQuery: String?,
    val mode: String?,
    val totalLatencyMs: Double?,
    val blipCaption: String?,
    val ocrPreview: String?,
    val ocrStatus: String?,
    val audioBytes: ByteArray,
)


@Composable
private fun VisionTestShellScreen() {
    val context = LocalContext.current
    val client = remember { OkHttpClient() }
    var serverUrl by remember { mutableStateOf("ws://10.0.2.2:8000/ws/vision") }
    var imageName by remember { mutableStateOf("Secilmedi") }
    var imageBytes by remember { mutableStateOf<ByteArray?>(null) }
    var audioName by remember { mutableStateOf("Yok") }
    var audioBytes by remember { mutableStateOf<ByteArray?>(null) }
    var statusText by remember { mutableStateOf("Hazir") }
    var isSending by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf<VisionUiResult?>(null) }

    DisposableEffect(Unit) {
        onDispose { client.dispatcher.executorService.shutdown() }
    }

    val imagePicker = rememberLauncherForActivityResult(OpenDocument()) { uri ->
        if (uri != null) {
            context.takePersistableUriPermissionSafe(uri)
            imageName = uri.lastPathSegment ?: "gorsel"
            imageBytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            statusText = if (imageBytes != null) "Gorsel yüklendi" else "Gorsel okunamadi"
        }
    }

    val audioPicker = rememberLauncherForActivityResult(OpenDocument()) { uri ->
        if (uri != null) {
            context.takePersistableUriPermissionSafe(uri)
            audioName = uri.lastPathSegment ?: "ses"
            audioBytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            statusText = if (audioBytes != null) "WAV yüklendi" else "WAV okunamadi"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "SAM Vision Test Shell",
            style = MaterialTheme.typography.headlineSmall,
            fontWeight = FontWeight.Bold,
        )
        Text("Emulator icin varsayilan adres: ws://10.0.2.2:8000/ws/vision")

        OutlinedTextField(
            value = serverUrl,
            onValueChange = { serverUrl = it },
            label = { Text("WebSocket URL") },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
        )

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(onClick = { imagePicker.launch(arrayOf("image/*")) }) {
                Text("Gorsel Sec")
            }
            Button(onClick = { audioPicker.launch(arrayOf("audio/wav", "audio/x-wav", "audio/*")) }) {
                Text("WAV Sec")
            }
            Button(onClick = {
                audioName = "Yok"
                audioBytes = null
            }) {
                Text("WAV Temizle")
            }
        }

        Text("Gorsel: $imageName")
        Text("WAV: $audioName")
        Text("Durum: $statusText")

        Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
            Button(
                enabled = !isSending && imageBytes != null,
                onClick = {
                    sendPayload(
                        client = client,
                        serverUrl = serverUrl,
                        payload = buildPayload(imageBytes!!, null),
                        onStarted = {
                            isSending = true
                            statusText = "Scene istegi gonderildi"
                            result = null
                        },
                        onFinished = { response ->
                            isSending = false
                            statusText = "Scene istegi tamamlandi"
                            result = response
                        },
                        onError = { error ->
                            isSending = false
                            statusText = error
                        },
                    )
                },
            ) {
                Text("Scene Gonder")
            }

            Button(
                enabled = !isSending && imageBytes != null && audioBytes != null,
                onClick = {
                    sendPayload(
                        client = client,
                        serverUrl = serverUrl,
                        payload = buildPayload(imageBytes!!, audioBytes),
                        onStarted = {
                            isSending = true
                            statusText = "Audio+image istegi gonderildi"
                            result = null
                        },
                        onFinished = { response ->
                            isSending = false
                            statusText = "Audio+image istegi tamamlandi"
                            result = response
                        },
                        onError = { error ->
                            isSending = false
                            statusText = error
                        },
                    )
                },
            ) {
                Text("Audio+Image Gonder")
            }
        }

        if (isSending) {
            CircularProgressIndicator()
        }

        result?.let { output ->
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "Sonuc",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
            )
            Text("Aciklama: ${output.description}")
            Text("Kullanici sorgusu: ${output.userQuery ?: "-"}")
            Text("Mod: ${output.mode ?: "-"}")
            Text("Toplam sure: ${output.totalLatencyMs ?: 0.0} ms")
            Text("BLIP caption: ${output.blipCaption ?: "-"}")
            Text("OCR preview: ${output.ocrPreview ?: "-"}")
            Text("OCR durum: ${output.ocrStatus ?: "-"}")
            Text("Ham JSON: ${output.rawJson}")

            Button(
                onClick = { playAudio(context, output.audioBytes) },
                enabled = output.audioBytes.isNotEmpty(),
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null)
                Text("MP3 Oynat")
            }
        }
    }
}


private fun buildPayload(imageBytes: ByteArray, audioBytes: ByteArray?): ByteArray {
    if (audioBytes == null) {
        return imageBytes
    }
    val prefix = ByteBuffer.allocate(4)
        .order(ByteOrder.LITTLE_ENDIAN)
        .putInt(audioBytes.size)
        .array()
    return prefix + audioBytes + imageBytes
}


private fun sendPayload(
    client: OkHttpClient,
    serverUrl: String,
    payload: ByteArray,
    onStarted: () -> Unit,
    onFinished: (VisionUiResult) -> Unit,
    onError: (String) -> Unit,
) {
    val request = Request.Builder().url(serverUrl).build()
    val audioBuffer = ByteArrayOutputStream()
    var metadata: JSONObject? = null
    var streaming = false
    val finished = AtomicBoolean(false)

    fun complete(result: VisionUiResult, socket: WebSocket) {
        if (finished.compareAndSet(false, true)) {
            socket.close(1000, "done")
            onFinished(result)
        }
    }

    fun fail(message: String, socket: WebSocket?) {
        if (finished.compareAndSet(false, true)) {
            socket?.close(1011, "error")
            onError(message)
        }
    }

    onStarted()

    client.newWebSocket(request, object : WebSocketListener() {
        override fun onOpen(webSocket: WebSocket, response: Response) {
            webSocket.send(payload.toByteString())
        }

        override fun onMessage(webSocket: WebSocket, text: String) {
            val json = JSONObject(text)
            when (json.optString("status")) {
                "ping" -> return
                "error" -> fail(json.optString("message", "Bilinmeyen hata"), webSocket)
                "busy" -> fail("Sunucu mesgul, istek atlandi.", webSocket)
                "ok" -> {
                    metadata = json
                    streaming = json.optBoolean("audio_streaming", false)
                }
                "audio_end" -> {
                    val meta = metadata ?: return fail("Metadata alinamadi.", webSocket)
                    complete(buildResult(meta, audioBuffer.toByteArray()), webSocket)
                }
            }
        }

        override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
            audioBuffer.write(bytes.toByteArray())
            val meta = metadata
            if (meta != null && !streaming) {
                complete(buildResult(meta, audioBuffer.toByteArray()), webSocket)
            }
        }

        override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
            fail(t.message ?: "WebSocket basarisiz oldu.", webSocket)
        }
    })
}


private fun buildResult(metadata: JSONObject, audioBytes: ByteArray): VisionUiResult {
    val pipeline = metadata.optJSONObject("pipeline")
    return VisionUiResult(
        rawJson = metadata.toString(),
        description = metadata.optString("description"),
        userQuery = metadata.optString("user_query").takeIf { it.isNotBlank() && it != "null" },
        mode = metadata.optString("mode"),
        totalLatencyMs = metadata.optDouble("total_latency_ms"),
        blipCaption = pipeline?.optJSONObject("blip")?.optString("caption"),
        ocrPreview = pipeline?.optJSONObject("ocr")?.optString("text_preview"),
        ocrStatus = metadata.optString("ocr_status").takeIf { it.isNotBlank() },
        audioBytes = audioBytes,
    )
}


private fun playAudio(context: Context, audioBytes: ByteArray) {
    if (audioBytes.isEmpty()) return
    val tempFile = File(context.cacheDir, "vision_result.mp3")
    tempFile.writeBytes(audioBytes)
    val mediaPlayer = MediaPlayer()
    mediaPlayer.setDataSource(tempFile.absolutePath)
    mediaPlayer.setOnCompletionListener { player ->
        player.release()
    }
    mediaPlayer.prepare()
    mediaPlayer.start()
}


private fun Context.takePersistableUriPermissionSafe(uri: Uri) {
    runCatching {
        contentResolver.takePersistableUriPermission(
            uri,
            Intent.FLAG_GRANT_READ_URI_PERMISSION,
        )
    }
}
