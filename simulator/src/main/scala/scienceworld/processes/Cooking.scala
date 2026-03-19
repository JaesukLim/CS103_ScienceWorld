package scienceworld.processes

import scienceworld.objects.substance.food.{BakedPotato, Bread, BurntFood, ToastedMarshmallow}
import scienceworld.struct.EnvObject

object Cooking {
  case class TransformationRule(triggerName:String, minTempC:Double, ticksRequired:Int, mkResult:() => EnvObject)

  private val cookRules = Array(
    TransformationRule("potato", 120.0, 4, () => new BakedPotato()),
    TransformationRule("dough", 140.0, 4, () => new Bread()),
    TransformationRule("marshmallow", 80.0, 2, () => new ToastedMarshmallow())
  )

  private val overheatRules = Array(
    TransformationRule("baked potato", 220.0, 4, () => new BurntFood("potato")),
    TransformationRule("bread", 230.0, 4, () => new BurntFood("bread")),
    TransformationRule("toasted marshmallow", 160.0, 2, () => new BurntFood("marshmallow"))
  )

  def cookingTick(obj:EnvObject): Unit = {
    if (obj.isDeleted() || obj.propMaterial.isEmpty) return

    if (applyMatchingRule(obj, cookRules, isOverheatRule = false)) return
    applyMatchingRule(obj, overheatRules, isOverheatRule = true)
  }

  private def applyMatchingRule(obj:EnvObject, rules:Array[TransformationRule], isOverheatRule:Boolean): Boolean = {
    val maybeRule = rules.find(rule => rule.triggerName == obj.name.toLowerCase)
    if (maybeRule.isEmpty) return false

    val rule = maybeRule.get
    if (obj.propMaterial.get.temperatureC < rule.minTempC) return false

    if (isOverheatRule) {
      obj.propMaterial.get.overheatingTicks += 1
      if (obj.propMaterial.get.overheatingTicks < rule.ticksRequired) return false
    } else {
      obj.propMaterial.get.cookingTicks += 1
      if (obj.propMaterial.get.cookingTicks < rule.ticksRequired) return false
    }

    replaceObject(oldObj = obj, newObj = rule.mkResult())
    true
  }

  private def replaceObject(oldObj:EnvObject, newObj:EnvObject): Unit = {
    val container = oldObj.getContainer()
    if (container.isEmpty) return

    if (oldObj.propMaterial.isDefined && newObj.propMaterial.isDefined) {
      newObj.propMaterial.get.temperatureC = oldObj.propMaterial.get.temperatureC
    }

    oldObj.delete()
    container.get.addObject(newObj)
  }
}
