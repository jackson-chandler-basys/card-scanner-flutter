package com.basys.card_scanner.scanner_core

import android.annotation.SuppressLint
import android.os.CountDownTimer
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.basys.card_scanner.SingleFrameCardScanner
import com.basys.card_scanner.logger.debugLog
import com.basys.card_scanner.onCardScanFailed
import com.basys.card_scanner.onCardScanned
import com.basys.card_scanner.scanner_core.models.CardDetails
import com.basys.card_scanner.scanner_core.models.CardScannerOptions
import com.basys.card_scanner.scanner_core.optimizer.CardDetailsScanOptimizer
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.core.Core

class CardScanner(
    private val scannerOptions: CardScannerOptions,
    private val onCardScanned: onCardScanned,
    private val onCardScanFailed: onCardScanFailed
) : ImageAnalysis.Analyzer {

    private val singleFrameCardScanner: SingleFrameCardScanner = SingleFrameCardScanner(scannerOptions)
    private val cardDetailsScanOptimizer: CardDetailsScanOptimizer = CardDetailsScanOptimizer(scannerOptions)
    private var scanCompleted: Boolean = false

    init {
        if (!OpenCVLoader.initLocal()) {
            Log.e("OpenCV", "OpenCV initialization failed!")
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully!")
        }

        if (scannerOptions.cardScannerTimeOut > 0) {
            val timer = object : CountDownTimer((scannerOptions.cardScannerTimeOut * 1000).toLong(), 1000) {
                override fun onTick(millisUntilFinished: Long) {}

                override fun onFinish() {
                    debugLog("Card scanner timeout reached", scannerOptions)
                    val cardDetails = cardDetailsScanOptimizer.getOptimalCardDetails()
                    if (cardDetails != null) {
                        finishCardScanning(cardDetails)
                    } else {
                        onCardScanFailed()
                    }
                    debugLog("Finishing card scan with card details : $cardDetails", scannerOptions)
                }
            }
            timer.start()
        }
    }

    companion object {
        private const val TAG = "TextRecognitionProcess"
    }

    @OptIn(ExperimentalGetImage::class)
    @SuppressLint("UnsafeExperimentalUsageError")
    override fun analyze(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val image = InputImage.fromMediaImage(mediaImage, 90)
            val preprocessedImage = preprocessImage(image)

            val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

            recognizer.process(preprocessedImage)
                .addOnSuccessListener { visionText ->
                    if (scanCompleted) return@addOnSuccessListener
                    val cardDetails = singleFrameCardScanner.scanSingleFrame(visionText)
                        ?: return@addOnSuccessListener

                    if (scannerOptions.enableDebugLogs) {
                        debugLog("----------------------------------------------------", scannerOptions)
                        for (block in visionText.textBlocks) {
                            debugLog("visionText: TextBlock ============================", scannerOptions)
                            debugLog("visionText : ${block.text}", scannerOptions)
                        }
                        debugLog("----------------------------------------------------", scannerOptions)
                        debugLog("Card details : $cardDetails", scannerOptions)
                    }
                    cardDetailsScanOptimizer.processCardDetails(cardDetails)
                    if (cardDetailsScanOptimizer.isReadyToFinishScan()) {
                        finishCardScanning(cardDetailsScanOptimizer.getOptimalCardDetails()!!)
                    }
                }
                .addOnFailureListener { e ->
                    debugLog("Error : $e", scannerOptions)
                }
                .addOnCompleteListener {
                    imageProxy.close()
                }
        }
    }

    private fun preprocessImage(inputImage: InputImage): InputImage {
    
    val bitmap = inputImage.bitmapInternal ?: return inputImage
    val mat = Mat()
    Utils.bitmapToMat(bitmap, mat)

    Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2GRAY)

    val kernel = Mat(3, 3, CvType.CV_16S)
    kernel.put(0, 0, -1.0, -1.0, -1.0, -1.0, 9.0, -1.0, -1.0, -1.0, -1.0)
    Imgproc.filter2D(mat, mat, mat.depth(), kernel)

    Core.convertScaleAbs(mat, mat, 1.2, 10.0)

    Core.add(mat, Scalar(50.0), mat)

    val mask = Mat(mat.size(), CvType.CV_32F, Scalar(1.0))
    val center = Point(mat.width() / 2.0, mat.height() / 2.0)
    val radius = (min(mat.width(), mat.height()) / 2).toDouble()
    Imgproc.circle(mask, center, radius.toInt(), Scalar(0.0), -1)

    Core.multiply(mat, mask, mat)

    val blurredMat = Mat()
    Imgproc.GaussianBlur(mat, blurredMat, Size(0.0, 0.0), 10.0)
    Core.addWeighted(mat, 1.5, blurredMat, -0.5, 0.0, mat)

    val preprocessedBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888)
    Utils.matToBitmap(mat, preprocessedBitmap)

    return InputImage.fromBitmap(preprocessedBitmap, inputImage.rotationDegrees)
}

    private fun finish0CardScanning(cardDetails: CardDetails) {
        debugLog("OPTIMAL Card details : $cardDetails", scannerOptions)
        scanCompleted = true
        onCardScanned(cardDetails)
    }
}
