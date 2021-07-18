package com.example.talktomyhand

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.example.talktomyhand.ml.MobilenetV110224Quant
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var imgView: ImageView
    lateinit var bitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imgView = findViewById(R.id.imageView)
        val textView: TextView = findViewById(R.id.textView2)

        val selectButton: Button = findViewById(R.id.button)

        selectButton.setOnClickListener {

            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"

            startActivityForResult(intent, 100)

        }

        val predictButton: Button = findViewById(R.id.button2)
        predictButton.setOnClickListener{

            val resizedImage: Bitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            val model = MobilenetV110224Quant.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            val tensorBuffer = TensorImage.fromBitmap(resizedImage)
            val byteBuffer = tensorBuffer.buffer
            inputFeature0.loadBuffer(byteBuffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer

            textView.setText(outputFeature0.floatArray[10].toString())

            // Releases model resources if no longer used.
            model.close()

        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        val uri:Uri? = data?.data
        // bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
        try {
            uri?.let {
                if(Build.VERSION.SDK_INT < 28) {
                    val bitmap = MediaStore.Images.Media.getBitmap(
                            this.contentResolver,
                            uri
                    )
                    imgView.setImageBitmap(bitmap)
                } else {
                    val source = ImageDecoder.createSource(this.contentResolver, uri)
                    val bitmap = ImageDecoder.decodeBitmap(source)
                    imgView.setImageBitmap(bitmap)
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

    }

    private fun getMax(arr:FloatArray): Int {
        val maximum = arr.maxOrNull() ?:0
        return arr.indexOfFirst { it == maximum }
    }
}