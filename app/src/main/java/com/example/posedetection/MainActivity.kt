package com.example.posedetection

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import android.view.TextureView
import android.widget.ImageView
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.example.posedetection.ml.AutoModel1
import com.example.posedetection.ml.AutoModel4
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {
    private lateinit var paint: Paint
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var model: AutoModel4
    private lateinit var bitmap: Bitmap
    private lateinit var imageView: ImageView
    private lateinit var handler: Handler
    private lateinit var handlerThread: HandlerThread
    private lateinit var textureView: TextureView
    private lateinit var cameraManager: CameraManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        getPermission()

        // Initialize model and image processor
        model = AutoModel4.newInstance(this)
        imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(192, 192, ResizeOp.ResizeMethod.BILINEAR))
            .build()

        imageView = findViewById(R.id.iamgeview)
        textureView = findViewById(R.id.textureview)
        cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
        handlerThread = HandlerThread("videothread")
        handlerThread.start()
        handler = Handler(handlerThread.looper)

        paint = Paint().apply {
            color = Color.RED
            style = Paint.Style.FILL
        }

        textureView.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) {
                openCamera()
            }

            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) {}

            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                return false
            }

            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) {
                bitmap = textureView.bitmap ?: return
                val tensorImage = TensorImage(DataType.UINT8)
                tensorImage.load(bitmap)
                val processedImage = imageProcessor.process(tensorImage)

                // Create TensorBuffer with the shape expected by the AutoModel4 model
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 192, 192, 3), DataType.UINT8)
                inputFeature0.loadBuffer(processedImage.buffer)

                // Run model inference
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

                // Draw results on canvas
                val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                val canvas = Canvas(mutableBitmap)
                val height = bitmap.height
                val width = bitmap.width
                var x = 0
                while (x <= 49) {
                    if (outputFeature0[x + 2] > 0.45) {
                        canvas.drawCircle(outputFeature0[x + 1] * width, outputFeature0[x] * height, 10f, paint)
                    }
                    x += 3
                }

                imageView.setImageBitmap(mutableBitmap)
            }
        }
    }

    private fun openCamera() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            return
        }
        cameraManager.openCamera(cameraManager.cameraIdList[0], object : CameraDevice.StateCallback() {
            override fun onOpened(cameraDevice: CameraDevice) {
                val captureRequest = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW)
                val surface = Surface(textureView.surfaceTexture)
                captureRequest.addTarget(surface)
                cameraDevice.createCaptureSession(listOf(surface), object : CameraCaptureSession.StateCallback() {
                    override fun onConfigured(session: CameraCaptureSession) {
                        session.setRepeatingRequest(captureRequest.build(), null, handler)
                    }

                    override fun onConfigureFailed(session: CameraCaptureSession) {
                        // Handle failure
                    }
                }, handler)
            }

            override fun onDisconnected(cameraDevice: CameraDevice) {
                // Handle disconnection
            }

            override fun onError(cameraDevice: CameraDevice, error: Int) {
                // Handle error
            }
        }, handler)
    }

    private fun getPermission() {
        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 101)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
            getPermission()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close() // Releases model resources
    }
}
